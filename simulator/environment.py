#===================================================================================================#
#                                      AdAS Environment                                             #
#    Last Modification: 13.03.2020                                         Mauricio Fadel Argerich  #
#===================================================================================================#

import cloudpickle
from collections import OrderedDict, defaultdict
from gym import spaces
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sys
import time

sys.path.append('scripts/simulator/')
sys.path.append('scripts/utils/')

from entities import AdASDevice, AdASCpu
from simulator import AdASSimulator
    

class AdASEnvironment:

    def __init__(self, simulator, latency_target, inputs, sample_inputs = False,
                 n_steps = None, state_mask = None):
        """
        Creates an AdASEnvironment with a simulator, a latency target and a set 
        of inputs.
        Keyword arguments:
        - simulator: an AdASSimulator class, with a profile loaded.
        - latency_target: maximum end-to-end latency allowed for application.
        - inputs: an array of AdASIO that will be used one per step.
        - sample_inputs: True if inputs should be sampled, False if should be used 
                         as list.
        - n_steps: number of steps before environment finishes, if None environment
                   never finishes.
        - state_mask: defines array to be returned as state. If None, all are 
                      returned. Options: 'available_cpu', 'latency', 
                      'params', 'utility', 'input'.
                      Note: params will add one element per parameter to the 
                      state array.
        """
        self.simulator = simulator
        self.latency_target = latency_target
        self.inputs = inputs
        self.sample_inputs = sample_inputs
        self.n_steps = n_steps
        self.set_combinations(simulator.profile.pipeline_data)
        
        # State is set in update_state().
        self.__state = OrderedDict()
        self.update_state(None, None)
        
        # Set state mask. If None, full state will be returned.
        if state_mask == None:
            self.state_mask = list(self.__state.keys())
        else:
            self.state_mask = []
            for s in state_mask:
                if s == 'params':
                    self.state_mask.extend(self.params_names)
                elif s in self.__state.keys():
                    self.state_mask.append(s)
                else:
                    raise ValueError('Unknown element in state mask! Available options: ' + list(self.__state.keys()))

        # CPU, latency and one discrete value sorted by ascending utility for each param.
        self.observation_space = spaces.Box(low = self.get_low_state(), 
                                            high = self.get_high_state(), 
                                            dtype = np.float32)
        # Actions are each combination of params to use.
        self.action_space = spaces.Discrete(len(self.params_values_comb))
        
        
    def get_low_state(self):
        """
        Returns a numpy array with the minimum value of each state element in self.state_mask.
        """
        # cpu_availability, latency, params, utility.
        low_state = []
        for m in self.state_mask:
            if m in self.params_names:
                # All parameters are in range [0,len(values)).
                low_state.append(0) 
            elif m == 'available_cpu':
                low_state.append(0)
            elif m == 'last_latency':
                low_state.append(0)
            elif m == 'utility':
                low_state.append(self.min_utility)
            else:
                # Not gonna happen because it's already checked in __init__ ;)
                pass
        return np.array(low_state)
    
    def get_high_state(self):
        """
        Returns a numpy array with the maximum value of each state element in self.state_mask.
        """
        high_state = []
        for m in self.state_mask:
            if m in self.params_names:
                # All parameters are in range [0,len(values)).
                high_state.append((len(self.params_names_values.get(m))-1)) 
            elif m == 'available_cpu':
                high_state.append(1)
            elif m == 'last_latency':
                high_state.append(np.Infinity)
            elif m == 'utility':
                high_state.append(self.max_utility)
            else:
                # Not gonna happen because it's already checked in __init__ ;)
                pass
        return np.array(high_state)
    
    
    def get_input(self):
        """
        Returns the input for current step.
        """
        if self.sample_inputs:
            if self.__state.get('input') == None or abs(np.random.randn()) < 0.1:
                new_input = np.random.choice(self.inputs)
            else:
                new_input = self.__state.get('input')
        else:
            new_input = self.inputs[self.iter]
            
        return new_input
        
        
    def update_state(self, old_state, action):
        """
        Takes old state and action and updates self.state.
        Keyword arguments:
        - old_state: previous state.
        - action: action taken by agent.
        """
        
        if old_state == None and action == None:
            self.__state['available_cpu'] =  1
            self.__state['latency'] = 0.9
            for pn in self.params_names:
                self.__state[pn] = 0
            self.__state['utility'] = self.max_utility
            self.__state['input'] = None
            # Reset step number.
            self.iter = 0
        else:
            # Simulate new cpu availability and set it..
            old_cpu = old_state.get('available_cpu')
            if abs(np.random.randn()) < 0.1:
                new_cpu = old_cpu + np.random.randn()/10
                if new_cpu > 1.0:
                    new_cpu = 1.0
                elif new_cpu < 0.3:
                    new_cpu = old_cpu
            else:
                new_cpu = old_cpu
            self.__state['available_cpu'] = new_cpu

            # Get current params.
            params_to_use = self.get_params_values(self.params_values_comb[action])
            
            # Get and set current input.
            self.__state['input'] = self.get_input()

            # Calculate and set current latency using simulator.
            exec_res = self.simulator.sim(self.__state.get('input'),
                                          params_to_use, self.__state['available_cpu'])
            # Sum latencies for all functions in pipelines and set it.
            self.__state['latency'] = np.sum(exec_res.get('latency')) 

            # Set params (here only because of order).
            for pn, v in params_to_use.items():
                self.__state[pn] = self.get_value_ix(pn, v)

            # Get and set current utility from simulator.
            self.__state['utility'] = self.simulator.get_utility(params_to_use)
            
            # Update step number.
            self.iter += 1
    
    
    def set_combinations(self, pipeline):
        """
        Creates all the possible combinations of values for the different parameters.
        It sets: max and min utilities, paramsn_names, params_values_comb and params_names_values.
        """
        params_names = []
        params_values = []
        params_names_values = {}
        for f in pipeline:
            for param_name in sorted(f.params.keys()):
                params_names.append(param_name)
                params_values.append(list(f.params.get(param_name).keys()))
                params_names_values[param_name] = f.params.get(param_name)

        # Get combinations utilities and filter out the not valid ones.
        comb_dict = {}
        for c in list(itertools.product(*params_values)):
            #try:
            exec_res = self.simulator.sim(self.inputs[0], self.get_params_values(c, params_names), 1.0)
            comb_dict[c] = self.simulator.get_utility(self.get_params_values(c, params_names))
            #except:
            #    pass

        sorted_comb = sorted(comb_dict, key=comb_dict.__getitem__)
        self.max_utility = comb_dict.get(sorted_comb[len(sorted_comb) - 1])
        self.min_utility = comb_dict.get(sorted_comb[0])
        self.params_names = params_names
        self.params_values_comb = sorted_comb
        self.params_names_values = params_names_values
        

    def get_params_values(self, comb, params_names = None):
        """
        Return the values for the parameters of a given combination.
        Keyword arguments:
        - comb: an int indicating the combination used.
        - params_names: the parameters to be included in result. If None returns all.
        """
        if params_names == None:
            # If params_names is None, return values for all params.
            params_names = self.params_names
        params_values = {}
        for p_i in range(len(comb)):
            params_values[params_names[p_i]] = comb[p_i]
        return params_values
    
    
    def get_value_ix(self, param_name, value):
        """
        Returns the index of a value for a given parameter.
        Keyword arguments:
        - param_name: name of parameter.
        - value: value of the parameter for which the index is returned.
        """
        ix = 0
        sorted_values = {k: v for k, v in sorted(self.params_names_values.get(param_name).items(), 
                                                 key=lambda item: item[1])}
        for k, v in sorted_values.items():
            if k == value:
                return ix
            
            ix += 1
    
    def compute_reward(self):
        """
        Computes and returns the reward for a given step.
        """
        lat = self.__state.get('latency')
        util = self.__state.get('utility')
        if lat <= self.latency_target:
            return (util - self.min_utility)/(self.max_utility - self.min_utility)
        else:
            return (self.latency_target - lat) / self.latency_target

        
    def is_finished(self):
        """
        Checks the termination condition, returns True if run is finished.
        """
        if self.sample_inputs:
            if self.n_steps != None and self.iter >= self.n_steps:
                # Run finishes when n_steps is reached.
                return True
            else:
                # n_steps has not been reached yet or n_steps is None.
                return False
        else:
            if self.iter >= len(self.inputs):
                # Run finishes when all the inputs have been used.
                return True
            else:
                # All inputs have not been used yet.
                return False
    
    def get_info(self, a):
        """
        Creates dictionary to be returned as info in each step.
        Keyword arguments:
        - a: action taken in this step.
        """
        info = {}
        info['available_cpu'] = self.__state.get('available_cpu')
        info['latency'] = self.__state.get('latency')
        for pn, v in self.get_params_values(self.params_values_comb[a]).items():
            info[pn] = str(self.get_value_ix(pn, v)) + ': ' + str(v)
        info['utility'] = self.__state.get('utility')
        info['input'] = self.__state.get('input')
        info['step'] = self.iter
        
        return info
    
    def get_masked_state(self):
        """
        Returns a numpy array with an element for each state element in self.state_mask.
        """
        return np.array([self.__state.get(m) for m in self.state_mask])
        
    def step(self, a):
        """
        Keyword arguments:
        - a: is an int representing the combination of params to use.
        """
        # Check if environment has not finished yet, if so, raise an exception.
        if self.is_finished():
            raise Exception('Calling step when the environment has already finished!')
        
        # Calculate, simulate and set new state.
        self.update_state(self.__state, a)
        # Convert state into array and if mask is defined, mask it.
        masked_state = self.get_masked_state()
        reward = self.compute_reward()
        finished = self.is_finished()
        info = self.get_info(a)
        
        return masked_state, reward, finished, info

    def reset(self):
        """
        Reset environment state and return the masked state.
        """
        self.update_state(None, None)
        return self.get_masked_state()