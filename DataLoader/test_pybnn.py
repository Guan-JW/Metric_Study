from functools import partial
import logging
import os
import time
import tempfile
from typing import Union, Dict, Any

import numpy as np
from scipy import stats

import ConfigSpace as CS
import lasagne
from sgmcmc.theano_mcmc import SGHMCSampler
from sgmcmc.bnn.model import zero_mean_unit_var_normalization
from sgmcmc.utils import sharedX, floatX, shuffle
import theano

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util import rng_helper
from hpobench.benchmarks.ml.pybnn import _get_net, BayesianNeuralNetwork, BNNOnToyFunction, BNNOnBostonHousing, BNNOnProteinStructure, BNNOnYearPrediction


class BayesianNeuralNetwork_HPO(BayesianNeuralNetwork):

    def train(self, X, y, X_val, y_val, X_test, y_test, *args, **kwargs):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.

        """

        # Clear old samples
        start_time = time.time()

        self.net = self.get_net(n_inputs=X.shape[1])

        nll, mse = self.negative_log_likelihood(self.net, self.Xt, self.Yt, X.shape[0], self.weight_prior, self.variance_prior)
        params = lasagne.layers.get_all_params(self.net, trainable=True)

        seed = self.rng.randint(1, 100000)
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(seed)

        if self.sampling_method == "sghmc":
            self.sampler = SGHMCSampler(rng=srng, precondition=self.precondition, ignore_burn_in=False)
        elif self.sampling_method == "sgld":
            self.sampler = SGLDSampler(rng=srng, precondition=self.precondition)
        else:
            logging.error("Sampling Strategy % does not exist!" % self.sampling_method)

        self.compute_err = theano.function([self.Xt, self.Yt], [mse, nll])
        self.single_predict = theano.function([self.Xt], lasagne.layers.get_output(self.net, self.Xt))

        self.samples.clear()

        if self.normalize_input:
            self.X, self.x_mean, self.x_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X

        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        self.sampler.prepare_updates(nll, params, self.l_rate, mdecay=self.mdecay,
                                     inputs=[self.Xt, self.Yt], scale_grad=X.shape[0])

        logging.info("Starting sampling")

        # Check if we have enough data points to form a minibatch
        # otherwise set the batchsize equal to the number of input points
        if self.X.shape[0] < self.bsize:
            self.bsize = self.X.shape[0]
            logging.error("Not enough datapoint to form a minibatch. "
                          "Set the batchsize to {}".format(self.bsize))

        i = 0
        self.train_losses = []
        self.valid_losses = []
        self.test_losses = []
        self.test_loss = 0
        while i < self.n_iters and len(self.samples) < self.n_nets:
            if self.X.shape[0] == self.bsize:
                start = 0
            else:
                start = np.random.randint(0, self.X.shape[0] - self.bsize)

            xmb = floatX(self.X[start:start + self.bsize])
            ymb = floatX(self.y[start:start + self.bsize, None])

            if i < self.burn_in:
                _, nll_value = self.sampler.step_burn_in(xmb, ymb)
            else:
                _, nll_value = self.sampler.step(xmb, ymb)

            if i % 512 == 0 and i <= self.burn_in:
                total_err, total_nll = self.compute_err(floatX(self.X), floatX(self.y).reshape(-1, 1))
                t = time.time() - start_time
                logging.info("Iter {:8d} : NLL = {:11.4e} MSE = {:.4e} "
                             "Time = {:5.2f}".format(i, float(total_nll),
                             float(total_err), t))

            if 1:
                train_err, _ = self.compute_err(floatX(X), floatX(y).reshape(-1, 1))
                self.train_losses.append(train_err)

                val_err, _ = self.compute_err(floatX(X_val), floatX(y_val).reshape(-1, 1))
                self.valid_losses.append(val_err)

                test_err, _ = self.compute_err(floatX(X_test), floatX(y_test).reshape(-1, 1))
                self.test_losses.append(test_err)

            i += 1

        self.is_trained = True
    
    # Defined by Jiawei
    def get_losses(self, X_train, y_train, X_val, y_val, X_test, y_test):
        return self.train_losses, self.valid_losses, self.test_losses
    

class BNNOnToyFunction_HPO(BNNOnToyFunction):
    
    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a bayesian neural network with 3 layers on the train and valid data split and evaluates it on the test
        split.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the pyBNN model
        fidelity: Dict, None
            budget : int [500 - 10000]
                number of epochs to train the model
            Fidelity parameters for the pyBNN model, check get_fidelity_space(). Uses default (max) value if None.

            Note: The fidelity should be here the max budget (= 10000). By leaving this field empty, the maximum budget
            will be used by default.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        start = time.time()

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        np.random.seed(self.rng.randint(1, 10000))

        # `burn_in_steps` must be at least 1, otherwise, theano will raise an RuntimeError. (Actually, the definition of
        # the config space allows as lower limit a zero. In this case, set the number of steps to 1.)
        
        burn_in_steps = max(1, int(configuration['burn_in'] * fidelity['budget']))

        net = partial(_get_net,
                      n_units_1=configuration['n_units_1'],
                      n_units_2=configuration['n_units_2'])

        model = BayesianNeuralNetwork_HPO(sampling_method="sghmc",
                                      get_net=net,
                                      l_rate=configuration['l_rate'],
                                      mdecay=configuration['mdecay'],
                                      burn_in=burn_in_steps,
                                      n_iters=fidelity['budget'],
                                      precondition=True,
                                      normalize_input=True,
                                      normalize_output=True,
                                      rng=self.rng)

        model.train(self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets)

        train_losses, valid_losses, test_losses = model.get_losses(self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets)

        cost = time.time() - start
        return {'train_losses': train_losses,
                'valid_losses': valid_losses,
                'test_losses': test_losses,
                'cost': cost,
                'info': {'fidelity': fidelity}}


class BNNOnBostonHousing_HPO(BNNOnBostonHousing):
    
    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a bayesian neural network with 3 layers on the train and valid data split and evaluates it on the test
        split.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the pyBNN model
        fidelity: Dict, None
            budget : int [500 - 10000]
                number of epochs to train the model
            Fidelity parameters for the pyBNN model, check get_fidelity_space(). Uses default (max) value if None.

            Note: The fidelity should be here the max budget (= 10000). By leaving this field empty, the maximum budget
            will be used by default.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        start = time.time()

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        np.random.seed(self.rng.randint(1, 10000))

        # `burn_in_steps` must be at least 1, otherwise, theano will raise an RuntimeError. (Actually, the definition of
        # the config space allows as lower limit a zero. In this case, set the number of steps to 1.)
        burn_in_steps = max(1, int(configuration['burn_in'] * fidelity['budget']))

        net = partial(_get_net,
                      n_units_1=configuration['n_units_1'],
                      n_units_2=configuration['n_units_2'])

        model = BayesianNeuralNetwork_HPO(sampling_method="sghmc",
                                      get_net=net,
                                      l_rate=configuration['l_rate'],
                                      mdecay=configuration['mdecay'],
                                      burn_in=burn_in_steps,
                                      n_iters=fidelity['budget'],
                                      precondition=True,
                                      normalize_input=True,
                                      normalize_output=True,
                                      rng=self.rng)

        model.train(self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets)

        train_losses, valid_losses, test_losses = model.get_losses(self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets)

        cost = time.time() - start
        return {'train_losses': train_losses,
                'valid_losses': valid_losses,
                'test_losses': test_losses,
                'cost': cost,
                'info': {'fidelity': fidelity}}


class BNNOnProteinStructure_HPO(BNNOnProteinStructure):
    
    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a bayesian neural network with 3 layers on the train and valid data split and evaluates it on the test
        split.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the pyBNN model
        fidelity: Dict, None
            budget : int [500 - 10000]
                number of epochs to train the model
            Fidelity parameters for the pyBNN model, check get_fidelity_space(). Uses default (max) value if None.

            Note: The fidelity should be here the max budget (= 10000). By leaving this field empty, the maximum budget
            will be used by default.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        start = time.time()

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        np.random.seed(self.rng.randint(1, 10000))

        # `burn_in_steps` must be at least 1, otherwise, theano will raise an RuntimeError. (Actually, the definition of
        # the config space allows as lower limit a zero. In this case, set the number of steps to 1.)
        
        burn_in_steps = max(1, int(configuration['burn_in'] * fidelity['budget']))

        net = partial(_get_net,
                      n_units_1=configuration['n_units_1'],
                      n_units_2=configuration['n_units_2'])

        model = BayesianNeuralNetwork_HPO(sampling_method="sghmc",
                                      get_net=net,
                                      l_rate=configuration['l_rate'],
                                      mdecay=configuration['mdecay'],
                                      burn_in=burn_in_steps,
                                      n_iters=fidelity['budget'],
                                      precondition=True,
                                      normalize_input=True,
                                      normalize_output=True,
                                      rng=self.rng)

        model.train(self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets)

        train_losses, valid_losses, test_losses = model.get_losses(self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets)

        cost = time.time() - start
        return {'train_losses': train_losses,
                'valid_losses': valid_losses,
                'test_losses': test_losses,
                'cost': cost,
                'info': {'fidelity': fidelity}}


class BNNOnYearPrediction_HPO(BNNOnYearPrediction):
    
    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None, **kwargs) -> Dict:
        """
        Trains a bayesian neural network with 3 layers on the train and valid data split and evaluates it on the test
        split.

        Parameters
        ----------
        configuration : Dict, CS.Configuration
            Configuration for the pyBNN model
        fidelity: Dict, None
            budget : int [500 - 10000]
                number of epochs to train the model
            Fidelity parameters for the pyBNN model, check get_fidelity_space(). Uses default (max) value if None.

            Note: The fidelity should be here the max budget (= 10000). By leaving this field empty, the maximum budget
            will be used by default.
        rng : np.random.RandomState, int, None,
            Random seed for benchmark. By default the class level random seed.

            To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : validation loss
            cost : time to train and evaluate the model
            info : Dict
                fidelity : used fidelities in this evaluation
        """
        start = time.time()

        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        np.random.seed(self.rng.randint(1, 10000))

        # `burn_in_steps` must be at least 1, otherwise, theano will raise an RuntimeError. (Actually, the definition of
        # the config space allows as lower limit a zero. In this case, set the number of steps to 1.)
        
        burn_in_steps = max(1, int(configuration['burn_in'] * fidelity['budget']))

        net = partial(_get_net,
                      n_units_1=configuration['n_units_1'],
                      n_units_2=configuration['n_units_2'])

        model = BayesianNeuralNetwork_HPO(sampling_method="sghmc",
                                      get_net=net,
                                      l_rate=configuration['l_rate'],
                                      mdecay=configuration['mdecay'],
                                      burn_in=burn_in_steps,
                                      n_iters=fidelity['budget'],
                                      precondition=True,
                                      normalize_input=True,
                                      normalize_output=True,
                                      rng=self.rng)

        model.train(self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets)

        train_losses, valid_losses, test_losses = model.get_losses(self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets)

        cost = time.time() - start
        return {'train_losses': train_losses,
                'valid_losses': valid_losses,
                'test_losses': test_losses,
                'cost': cost,
                'info': {'fidelity': fidelity}}
