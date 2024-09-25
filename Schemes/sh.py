import numpy as np
import pandas as pd
import re

import warnings
from random import random
from math import log, ceil
from time import time, ctime
import csv
import json
import ConfigSpace as CS

def min_max_scaling(values):
	min_val = min(values)
	max_val = max(values)
	range_val = max_val - min_val
	scaled_values = [(x - min_val) / range_val if range_val != 0 else 0 for x in values]
	return scaled_values

class SuccessiveHalving:
	
	def __init__( self, get_params_function, try_params_function, max_iter=81, eta=3, skip_first=0 ):
		self.get_params = get_params_function
		self.try_params = try_params_function
		
		self.max_iter = max_iter  	# maximum iterations per configuration
		self.eta = eta			# defines configuration downsampling rate (default = 3)
		self.skip_first = skip_first

		self.logeta = lambda x: log( x ) / log( self.eta )
		self.s_max = int( self.logeta( self.max_iter ))
		self.B = ( self.s_max + 1 ) * self.max_iter
		print(f"max_iter={self.max_iter}, s_max={self.s_max}, B={self.B}")
		# exit()

		# self.results = []	# list of dicts
		self.counter = 0
		# self.best_loss = np.inf
		# self.best_crt_val = np.inf	# minimization
		# self.best_counter = -1

		self.fixed_config_dict = dict()

	def __str__(self):
		return f"Max_iter_{self.max_iter}_eta_{self.eta}"

	def record_to_csv(self, results, record_file='./record.csv'):
		df = pd.DataFrame(results)
		df.to_csv(record_file, index=False)


	def run_fixed_configs( self, criteria = 'valid_accuracy', direction = None, dry_run = False ):
		# clear results
		results = []
		final_results = []

		# dealing with special criteria
		if criteria.startswith('wgh'):
			# Find all matches of integers in criteria
			pattern = r'\d+'
			matches = re.findall(pattern, criteria)
			# Extract the first two numbers
			if len(matches) >= 2:
				wgh1 = float(matches[0]) * 0.1
				wgh2 = float(matches[1]) * 0.1
				print(f"wgh1 = {wgh1}, wgh2 = {wgh2}")
			else:
				raise ValueError("Not enough numbers found in criteria.")
		
		s = self.s_max
		# initial number of configurations
		n = int( ceil( self.B / self.max_iter / ( s + 1 ) * self.eta ** s ))
			
		# initial number of iterations per config
		r = self.max_iter * self.eta ** ( -s )		

		# if rount zero, record configurations

		if f's_{s}' in self.fixed_config_dict:
			# get stored fixed configurations
			T = self.fixed_config_dict[f's_{s}']
		else:
			print(" !!!!! the first time !!!!! ")
			T = [ self.get_params() for i in range( n )] 
			self.fixed_config_dict[f's_{s}'] = T

		for i in range(self.skip_first, ( s + 1 )):	# changed from s + 1
				
			# Run each of the n configs for <iterations> 
			# and keep best (n_configs / eta) configurations
			
			n_configs = n * self.eta ** ( -i + self.skip_first )
			n_iterations = r * self.eta ** ( i )
				
			print( "\n*** {} configurations x {:.1f} iterations each".format( 
					n_configs, n_iterations ))
				
			criterias = []
			early_stops = []
				
			for t in T:
					
				self.counter += 1
				if dry_run:
					result = {criteria: random(), 'time': random()}
				else:
					result = self.try_params( n_iterations, t, criteria )		# <---
						
				assert( type( result ) == dict )
				assert( criteria in result )
				assert( 'time' in result )
				assert( 'test_accuracy' in result )
					
				seconds = result['time']
				print( "\n{} seconds.".format( seconds ))
					
				crt_val = result[criteria]
				criterias.append(crt_val)
					
				early_stop = result.get( 'early_stop', False )
				early_stops.append( early_stop )

				result['s'] = s
				result['counter'] = self.counter
				result['params'] = t
				result['n_iteration'] = n_iterations
					
				results.append( result )

				# last round of successive halving
				if i == (s):
					final_results.append(result)
				
			# select a number of best configurations for the next loop
			# filter out early stops, if any
			# print(f"criteria = {criteria}")
			if criteria.startswith('wgh'):
				# 1. Normalize both the lcr and second value
				lcrs = [cta[0] for cta in criterias]
				rvals = [cta[1] for cta in criterias]
				normed_lcrs = min_max_scaling(lcrs)
				normed_rvals = min_max_scaling(rvals)

				# 2. Combine with weights
				# 3. Replace the criterias, results, and final_results with new value
				# print(f"criterias.length = {len(criterias)}, results.length = {len(results)}, final_results.length = {len(final_results)}")
				# print(f"final_results = {final_results}")
				for cta_idx in range(len(criterias)):
					i = cta_idx + len(results) - len(criterias)
					wgh_val = wgh1 * normed_lcrs[cta_idx] + wgh2 * normed_rvals[cta_idx]
					results[i][criteria] = wgh_val
					criterias[cta_idx] = wgh_val

					if i >= len(results) - len(final_results):
						final_results[i - len(results) + len(final_results)][criteria] = wgh_val

			elif criteria.startswith('dyn_win'):
				sigs = np.array([cta[1] for cta in criterias])
				median_sig = np.median(sigs)
				mean_sig = np.mean(sigs)

				if mean_sig <= median_sig:	# most sigs are large
					print(f"Epoch = {n_iterations} -- Most sigs are LARGE !!!!")
					if direction == 'Max':	# Accuracy
						for cta_idx in range(len(criterias)):
							i = cta_idx + len(results) - len(criterias)
							val = criterias[cta_idx][0] + criterias[cta_idx][1]	# mu + sig
							results[i][criteria] = val
							criterias[cta_idx] = val

							if i >= len(results) - len(final_results):
								final_results[i - len(results) + len(final_results)][criteria] = val
					elif direction == 'Min':	# Loss
						for cta_idx in range(len(criterias)):
							i = cta_idx + len(results) - len(criterias)
							val = criterias[cta_idx][0] - criterias[cta_idx][1]	# mu - sig
							results[i][criteria] = val
							criterias[cta_idx] = val

							if i >= len(results) - len(final_results):
								final_results[i - len(results) + len(final_results)][criteria] = val
				else:	# most sigs are small
					print(f"Epoch = {n_iterations} -- Most sigs are SMALL !!!!")
					for cta_idx in range(len(criterias)):
						i = cta_idx + len(results) - len(criterias)
						val = criterias[cta_idx][0]	# mu
						results[i][criteria] = val
						criterias[cta_idx] = val

						if i >= len(results) - len(final_results):
							final_results[i - len(results) + len(final_results)][criteria] = val

			indices = np.argsort( criterias )
			if direction == 'Max':	# maximum
				# warnings.warn(f"criteria = {criteria}")
				indices = indices[::-1]
			T = [ T[i] for i in indices if not early_stops[i]]
			T = T[ 0:int( n_configs / self.eta )]
		
		# rank final result
		if direction == 'Max':	# maximum
			# warnings.warn(f"criteria = {criteria}")
			ranked = sorted(final_results, key=lambda x: x[criteria], reverse=True)
		elif direction == 'Min':
			ranked = sorted(final_results, key=lambda x: x[criteria])
		else:
			raise ValueError(f"Invalid direction '{direction}'.")
		
		# print(f"ranked = {ranked}")
		print(" ****** the best one ***** ")
		print(ranked[0])
		# append the best one to the last of rst
		results.append(ranked[0])

		# if criteria.startswith('wgh'):
		# 	exit(0)
		return results
	
	def get_fixed_config_dict(self, config_space):
		if not self.fixed_config_dict:
			raise ValueError("config_dict is empty.")
		# print(f'fixed_config_dict = {self.fixed_config_dict}')
		serialized_config_dict = dict()
		s = self.s_max
		T = []
		for config in self.fixed_config_dict[f's_{s}']:
			T.append(config.get_dictionary())
		serialized_config_dict[f's_{s}'] = T
		return serialized_config_dict
	
	def get_fixed_int_config_dict(self):
		if not self.fixed_config_dict:
			raise ValueError("config_dict is empty.")
		serialized_config_dict = dict()
		for s in reversed( range(self.skip_first, self.s_max + 1 )):
			T = []
			for config in self.fixed_config_dict[f's_{s}']:
				T.append(str(config))
			serialized_config_dict[f's_{s}'] = T
		return serialized_config_dict
	
	def load_fixed_config_dict(self, file_path, config_space):
		with open(file_path, "r") as json_file:
			loaded_configuration_dict = json.load(json_file)
		self.fixed_config_dict = dict()
		s = self.s_max
		T = []
		for config in loaded_configuration_dict[f's_{s}']:
			T.append(CS.Configuration(config_space, values=config))
		self.fixed_config_dict[f's_{s}'] = T

	def load_fixed_int_config_dict(self, file_path):
		print(f"{file_path=}")
		with open(file_path, "r") as json_file:
			loaded_configuration_dict = json.load(json_file)
		self.fixed_config_dict = dict()
		for s in reversed( range(self.skip_first, self.s_max + 1 )):
			T = []
			for config in loaded_configuration_dict[f's_{s}']:
				T.append(int(config))
			self.fixed_config_dict[f's_{s}'] = T

	def get_fixed_config_dict_lcbench(self):
		if not self.fixed_config_dict:
			raise ValueError("config_dict is empty.")

		return self.fixed_config_dict
	
	def load_fixed_config_dict_lcbench(self, file_path):
		self.fixed_config_dict = None
		with open(file_path, "r") as json_file:
			self.fixed_config_dict = json.load(json_file)
		
		if self.fixed_config_dict == None:
			raise ValueError("Error in loading configuration dictionary.")