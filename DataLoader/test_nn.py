import time
import warnings
import argparse
# from time import time
from copy import deepcopy
from typing import Union, Tuple, Dict
import ConfigSpace as CS
import numpy as np
import pandas as pd
from ConfigSpace.hyperparameters import Hyperparameter

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from sklearn.neural_network._multilayer_perceptron import _STOCHASTIC_SOLVERS
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer, AdamOptimizer
from sklearn.utils import gen_batches, check_random_state, shuffle, _safe_indexing
from sklearn.exceptions import ConvergenceWarning

from hpobench.util.rng_helper import get_rng
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util.openml_data_manager import get_openmlcc18_taskids
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark, NNBenchmarkBB, NNBenchmarkMF
from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark

# defined by jiawei
class MLPClassifier_HPO(MLPClassifier):
    def _fit_stochastic(self, X, y, X_val, y_val, X_test, y_test, activations, deltas, coef_grads,
                        intercept_grads, layer_units, incremental):

        if not incremental or not hasattr(self, '_optimizer'):
            params = self.coefs_ + self.intercepts_

            if self.solver == 'sgd':
                self._optimizer = SGDOptimizer(
                    params, self.learning_rate_init, self.learning_rate,
                    self.momentum, self.nesterovs_momentum, self.power_t)
            elif self.solver == 'adam':
                self._optimizer = AdamOptimizer(
                    params, self.learning_rate_init, self.beta_1, self.beta_2,
                    self.epsilon)

        # early_stopping in partial_fit doesn't make sense
        early_stopping = self.early_stopping and not incremental

        # added by jiawei
        tr_activations = [X] + [None] * (len(layer_units) - 1)
        val_activations = [X_val] + [None] * (len(layer_units) - 1)
        test_activations = [X_test] + [None] * (len(layer_units) - 1)

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            for it in range(self.max_iter):
                if self.shuffle:
                    # Only shuffle the sample indices instead of X and y to
                    # reduce the memory footprint. These indices will be used
                    # to slice the X and y.
                    sample_idx = shuffle(sample_idx,
                                         random_state=self._random_state)

                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    if self.shuffle:
                        X_batch = _safe_indexing(X, sample_idx[batch_slice])
                        y_batch = y[sample_idx[batch_slice]]
                    else:
                        X_batch = X[batch_slice]
                        y_batch = y[batch_slice]

                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = self._backprop(
                        X_batch, y_batch, activations, deltas,
                        coef_grads, intercept_grads)
                    accumulated_loss += batch_loss * (batch_slice.stop -
                                                      batch_slice.start)

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(grads)
                    break

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                # self.loss_curve_.append(self.loss_)


                # added by jiawei, calculate train loss
                tr_acts = self._forward_pass(tr_activations)
                loss_func_name = self.loss
                if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
                    loss_func_name = 'binary_log_loss'
                tr_loss = LOSS_FUNCTIONS[loss_func_name](y, tr_acts[-1])
                # Add L2 regularization term to loss
                values = np.sum(
                    np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
                tr_loss += (0.5 * self.alpha) * values / X.shape[0]
                self.loss_curve_.append(tr_loss)
                
                # added by jiawei, calculate valid loss
                v_acts = self._forward_pass(val_activations)
                loss_func_name = self.loss
                if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
                    loss_func_name = 'binary_log_loss'
                val_loss = LOSS_FUNCTIONS[loss_func_name](y_val, v_acts[-1])
                # Add L2 regularization term to loss
                values = np.sum(
                    np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
                val_loss += (0.5 * self.alpha) * values / X_val.shape[0]
                self.val_loss_curve_.append(val_loss)

            
                # added by jiawei, calculate valid loss
                t_acts = self._forward_pass(test_activations)
                test_loss = LOSS_FUNCTIONS[loss_func_name](y_test, t_acts[-1])
                # Add L2 regularization term to loss
                test_loss += (0.5 * self.alpha) * values / X_test.shape[0]
                self.test_loss_curve_.append(test_loss)

                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_,
                                                         self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(early_stopping, X_val, y_val)

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) "
                        "reached and the optimization hasn't converged yet."
                        % self.max_iter, ConvergenceWarning)
        
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    def _fit(self, X, y, X_val, y_val, X_test, y_test, incremental=False):
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        # Validate input parameters.
        self._validate_hyperparameters()
        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError("hidden_layer_sizes must be > 0, got %s." %
                             hidden_layer_sizes)
        
        X, y = self._validate_input(X, y, incremental)
        # added by jiawei
        X_val, y_val = self._validate_input(X_val, y_val, incremental)
        X_test, y_test = self._validate_input(X_test, y_test, incremental)

        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])

        # check random state
        self._random_state = check_random_state(self.random_state)

        if not hasattr(self, 'coefs_') or (not self.warm_start and not
                                           incremental):
            # First time training the model
            self._initialize(y, layer_units)

        # lbfgs does not support mini-batches
        if self.solver == 'lbfgs':
            batch_size = n_samples
        elif self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            if self.batch_size < 1 or self.batch_size > n_samples:
                warnings.warn("Got `batch_size` less than 1 or larger than "
                              "sample size. It is going to be clipped")
            batch_size = np.clip(self.batch_size, 1, n_samples)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [np.empty((n_fan_in_, n_fan_out_)) for n_fan_in_,
                      n_fan_out_ in zip(layer_units[:-1],
                                        layer_units[1:])]

        intercept_grads = [np.empty(n_fan_out_) for n_fan_out_ in
                           layer_units[1:]]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic(X, y, X_val, y_val, X_test, y_test, activations, deltas, coef_grads,
                                 intercept_grads, layer_units, incremental)

        # Run the LBFGS solver
        elif self.solver == 'lbfgs':
            self._fit_lbfgs(X, y, activations, deltas, coef_grads,
                            intercept_grads, layer_units)
        return self
    

    def fit(self, X, y, X_val, y_val, X_test, y_test):
        return self._fit(X, y, X_val, y_val, X_test, y_test, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))


class MLBenchmark_HPO(MLBenchmark):

    def _train_objective(self,
                         config: Dict,
                         fidelity: Dict,
                         shuffle: bool,
                         rng: Union[np.random.RandomState, int, None] = None,
                         evaluation: Union[str, None] = "valid"):

        if rng is not None:
            rng = get_rng(rng, self.rng)
        
        # initializing model
        model = self.init_model(config, fidelity, rng)

        train_X = self.train_X
        train_y = self.train_y
        train_idx = self.train_idx

        # shuffling data
        if shuffle:
            train_idx = self.shuffle_data_idx(train_idx, rng)
            train_X = train_X.iloc[train_idx]
            train_y = train_y.iloc[train_idx]

        # subsample here:
        # application of the other fidelity to the dataset that the model interfaces
        if self.lower_bound_train_size is None:
            self.lower_bound_train_size = (10 * self.n_classes) / self.train_X.shape[0]
            self.lower_bound_train_size = np.max((1 / 512, self.lower_bound_train_size))
        subsample = np.max((fidelity['subsample'], self.lower_bound_train_size))
        train_idx = self.rng.choice(
            np.arange(len(train_X)), size=int(
                subsample * len(train_X)
            )
        )
        # fitting the model with subsampled data
        start = time.time()
        model.fit(train_X[train_idx], train_y.iloc[train_idx], self.valid_X, self.valid_y, self.test_X, self.test_y)
        model_fit_time = time.time() - start

        return model, model_fit_time, model.loss_curve_, model.val_loss_curve_, model.test_loss_curve_

    # pylint: disable=arguments-differ
    @AbstractBenchmark.check_parameters
    def objective_function(self,
                           configuration: Union[CS.Configuration, Dict],
                           fidelity: Union[CS.Configuration, Dict, None] = None,
                           shuffle: bool = False,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """Function that evaluates a 'config' on a 'fidelity' on the validation set
        """
        model, model_fit_time, train_losses, val_losses, test_losses = self._train_objective(
            configuration, fidelity, shuffle, rng, evaluation="val"
        )

        test_scores = dict()
        test_score_cost = dict()
        for k, v in self.scorers.items():
            _start = time.time()
            test_scores[k] = v(model, self.test_X, self.test_y)
            test_score_cost[k] = time.time() - _start
        # test_loss = 1 - test_scores["acc"]
        test_loss = test_losses[-1]

        info = {
            'train_losses': train_losses,
            'valid_losses': val_losses,
            'test_losses': test_losses,
            'test_loss': test_loss,
            'test_accuracy': test_scores["acc"],
            'model_cost': model_fit_time,
            # storing as dictionary and not ConfigSpace saves tremendous memory
            'fidelity': fidelity,
            'config': configuration,
        }

        return info


# defined by jiawei
class NNBenchmark_HPO(MLBenchmark_HPO):
    def __init__(self,
                 task_id: int,
                 rng: Union[np.random.RandomState, int, None] = None,
                 valid_size: float = 0.33,
                 data_path: Union[str, None] = None):
        super(NNBenchmark_HPO, self).__init__(task_id, rng, valid_size, data_path)

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """Parameter space to be optimized --- contains the hyperparameters
        """
        cs = CS.ConfigurationSpace(seed=seed)

        cs.add_hyperparameters([
            CS.UniformIntegerHyperparameter(
                'depth', default_value=3, lower=1, upper=3, log=False
            ),
            CS.UniformIntegerHyperparameter(
                'width', default_value=64, lower=16, upper=1024, log=True
            ),
            CS.UniformIntegerHyperparameter(
                'batch_size', lower=4, upper=256, default_value=32, log=True
            ),
            CS.UniformFloatHyperparameter(
                'alpha', lower=10**-8, upper=1, default_value=10**-3, log=True
            ),
            CS.UniformFloatHyperparameter(
                'learning_rate_init', lower=10**-5, upper=1, default_value=10**-3, log=True
            )
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:

        fidelity_space = CS.ConfigurationSpace(seed=seed)
        fidelity_space.add_hyperparameters(
            # gray-box setting (multi-multi-fidelity) - iterations + data subsample
            NNBenchmark._get_fidelity_choices(iter_choice='variable', subsample_choice='variable')
        )
        return fidelity_space

    @staticmethod
    def _get_fidelity_choices(iter_choice: str, subsample_choice: str) -> Tuple[Hyperparameter, Hyperparameter]:

        fidelity1 = dict(
            fixed=CS.Constant('iter', value=243),
            variable=CS.UniformIntegerHyperparameter(
                'iter', lower=3, upper=243, default_value=243, log=False
            )
        )
        fidelity2 = dict(
            fixed=CS.Constant('subsample', value=1),
            variable=CS.UniformFloatHyperparameter(
                'subsample', lower=0.1, upper=1, default_value=1, log=False
            )
        )
        iter = fidelity1[iter_choice]
        subsample = fidelity2[subsample_choice]
        return iter, subsample

    def init_model(self, config: Union[CS.Configuration, Dict],
                   fidelity: Union[CS.Configuration, Dict, None] = None,
                   rng: Union[int, np.random.RandomState, None] = None):
        """ Function that returns the model initialized based on the configuration and fidelity
        """
        rng = self.rng if rng is None else rng

        if isinstance(config, CS.Configuration):
            config = config.get_dictionary()
        if isinstance(fidelity, CS.Configuration):
            fidelity = fidelity.get_dictionary()

        config = deepcopy(config)
        depth = config["depth"]
        width = config["width"]
        config.pop("depth")
        config.pop("width")
        hidden_layers = [width] * depth
        model = MLPClassifier_HPO(
            **config,
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=fidelity['iter'],  # a fidelity being used during initialization
            random_state=rng
        )
        return model

