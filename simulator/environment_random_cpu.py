#===================================================================================================#
#                                      AdAS Environment                                             #
#    Last Modification: 10.03.2020                                         Mauricio Fadel Argerich  #
#===================================================================================================#

import cloudpickle
from collections import OrderedDict, defaultdict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sys
import time
from gym import spaces

sys.path.append('scripts/simulator/')
sys.path.append('scripts/utils/')

from entities import MFADevice, MFACpu
from simulator import MFASimulator
    


class AdASEnvironment:

    def __init__(self, simulator, latency_target, inputs, state_mask = None):
        """
        Keyword arguments:
        - state_mask: defines array to be returned as state. If None, all are 
                      returned. Options: 'available_cpu', 'latency', 
                      'params', 'utility'.
                      Note: params will add one element per parameter to the 
                      state array.
        """
        self.simulator = simulator
        self.latency_target = latency_target
        self.inputs = inputs
        
        self.set_combinations(simulator.profile.pipeline_data)
        
        # State is set in update_state().
        self.iter = 0
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
                low_state.append(0)
            else:
                # Not gonna happen because it's already checked in __init__ ;)
                pass
        return np.array(low_state)
    
    def get_high_state(self):
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
                high_state.append(1)
            else:
                # Not gonna happen because it's already checked in __init__ ;)
                pass
        return np.array(high_state)
        
        
    def update_state(self, old_state, action):
        """
        Takes old state and action and returns new state.
        State: [available_cpu, last_latency, [value for each param], utility]
        """
        
        if old_state == None and action == None:
            self.__state['available_cpu'] =  1
            self.__state['latency'] = 0.9
            for pn in self.params_names:
                self.__state[pn] = 0
            self.__state['utility'] = 1
        
        else:
            # Simulate new cpu availability and set it..
            old_cpu = old_state.get('available_cpu')
            if abs(np.random.randn()) < 0.1:
                new_cpu = old_cpu + np.random.randn()/10
                if new_cpu > 1.0:
                    new_cpu = old_cpu
                elif new_cpu < 0.3:
                    new_cpu = old_cpu
            else:
                new_cpu = old_cpu
            self.__state['available_cpu'] = new_cpu

            # Get current params.
            params_to_use = self.get_params_values(self.params_values_comb[action])

            # Calculate current latency using simulator and set it.
            exec_res = self.simulator.sim('new_rp', self.inputs[self.iter],
                                          params_to_use, new_cpu)
            # Sum latencies for all functions in pipelines and set it.
            self.__state['latency'] = np.sum(exec_res.get('latency')) 

            # Set params (here only because of order).
            for pn, v in params_to_use.items():
                self.__state[pn] = self.get_value_ix(pn, v)

            # Get current utility from simulator.
            self.__state['utility'] = self.simulator.get_utility(params_to_use)
    
    
    def set_combinations(self, pipeline):
        params_names = []
        params_values = []
        params_names_values = {}
        for f in pipeline:
            for param_name in sorted(f.params_data.keys()):
                params_names.append(param_name)
                params_values.append(list(f.params_data.get(param_name).keys()))
                params_names_values[param_name] = f.params_data.get(param_name)

        # Get combinations utilities and filter out the not valid ones.
        comb_dict = {}
        for c in list(itertools.product(*params_values)):
            try:
                exec_res = self.simulator.sim('new_rp', self.inputs[0], self.get_params_values(c, params_names), 1.0)
                comb_dict[c] = self.simulator.get_utility(self.get_params_values(c, params_names))
            except:
                pass

        sorted_comb = sorted(comb_dict, key=comb_dict.__getitem__)
        self.max_utility = comb_dict.get(sorted_comb[len(sorted_comb) - 1])
        self.min_utility = comb_dict.get(sorted_comb[0])
        self.params_names = params_names
        self.params_values_comb = sorted_comb
        self.params_names_values = params_names_values
        

    def get_params_values(self, comb, params_names = None):
        if params_names == None:
            # If params_names is None, return values for all params.
            params_names = self.params_names
        params_values = {}
        for p_i in range(len(comb)):
            params_values[params_names[p_i]] = comb[p_i]
        return params_values
    
    
    def get_value_ix(self, param_name, value):
        ix = 0
        sorted_values = {k: v for k, v in sorted(self.params_names_values.get(param_name).items(), key=lambda item: item[1])}
        for k, v in sorted_values.items():
            if k == value:
                return ix
            
            ix += 1
    
    def compute_reward(self, lat, util):
        if lat <= self.latency_target:
            return (util - self.min_utility)/(self.max_utility - self.min_utility)
        else:
            return (self.latency_target - lat) / self.latency_target

        
    def is_finished(self):
        finished = False
        self.iter += 1
        if self.iter >= len(self.inputs):
            finished = True
        return finished
    
    def get_info(self, a):
        info = {}
        info['available_cpu'] = self.__state.get('available_cpu')
        info['latency'] = self.__state.get('latency')
        for pn, v in self.get_params_values(self.params_values_comb[a]).items():
            info[pn] = str(self.get_value_ix(pn, v)) + ': ' + str(v)
        info['utility'] = self.__state.get('utility')
        info['step'] = self.iter
        
        return info
    
    def get_masked_state(self):
        return np.array([self.__state.get(m) for m in self.state_mask])
        
    def step(self, a):
        # Action is combination of params to use.
        
        # Calculate, simulate and set new state.
        self.update_state(self.__state, a)
        # Convert state into array and if mask is defined, mask it.
        masked_state = self.get_masked_state()
        
        reward = self.compute_reward(self.__state.get('latency'), self.__state.get('utility'))
        finished = self.is_finished()
        info = self.get_info(a)
        
        return masked_state, reward, finished, info

    def reset(self):
        self.iter = 0
        self.update_state(None, None)
        return self.get_masked_state()
    
class MFAEnvironmentSimpleStates:

    def __init__(self, simulator, latency_target, inputs, cpu_steps):
        # cpu_steps is not used.
        self.simulator = simulator
        self.latency_target = latency_target
        self.inputs = inputs
        self.available_cpu = self.get_new_available_cpu(0.3 + abs(np.random.randn())*0.7)

        self.set_combinations(simulator.profile.pipeline_data)
        self.__state_map = self.build_state_map()
        self.observation_space = len(self.__state_map.keys())   # latency * comb_params
        self.action_space = len(self.params_values_comb)
        self.reset()
        
    def get_new_available_cpu(self, old_cpu):
        if abs(np.random.randn()) < 0.1:
            new_cpu = old_cpu + np.random.randn()/10
            if new_cpu > 1.0:
                new_cpu = old_cpu
            elif new_cpu < 0.3:
                new_cpu = old_cpu
        else:
            new_cpu = old_cpu
        
        return new_cpu

    def build_state_map(self):
        state_map = {}
        i = 0
        for c in list(itertools.product(range(3), range(len(self.params_values_comb)))):
            state_map[c] = i
            i += 1

        return state_map

    def set_combinations(self, pipeline_data):
        params_names = []
        params_values = []
        for f in pipeline_data:
            for param_name in f.params_data.keys():
                params_names.append(param_name)
                params_values.append(list(f.params_data.get(param_name).keys()))

        # Get combinations utilities and filter out the not valid ones.
        comb_dict = {}
        for c in list(itertools.product(*params_values)):
            try:
                exec_res = self.simulator.sim('new_rp', self.inputs[0], self.get_params_values(c, params_names), 1.0)
                comb_dict[c] = self.simulator.get_utility(self.get_params_values(c, params_names))
            except:
                pass

        sorted_comb = sorted(comb_dict, key=comb_dict.__getitem__)
        self.max_utility = comb_dict.get(sorted_comb[len(sorted_comb) - 1])
        self.min_utility = comb_dict.get(sorted_comb[0])
        self.params_names = params_names
        self.params_values_comb = sorted_comb

    def get_params_values(self, comb, params_names = None):
        if params_names == None:
            params_names = self.params_names
        params_values = {}
        for p_i in range(len(comb)):
            params_values[params_names[p_i]] = comb[p_i]

        return params_values

    def discretize_lat(self, lat):
        s = lat / self.latency_target
        if s < 0.8:
            return 0
        elif s < 1.0:
            return 1
        else:
            return 2

    def encode_state(self, lat, params_comb):
        lat_state = self.discretize_lat(lat)
        return self.__state_map.get((lat_state, params_comb))

    def compute_reward(self, lat, util):
        if lat <= self.latency_target:
            return (util - self.min_utility)/(self.max_utility - self.min_utility)
        else:
            return (self.latency_target - lat) / self.latency_target

    def step(self, a):
        self.available_cpu = self.get_new_available_cpu(self.available_cpu)
        params_to_use = self.get_params_values(self.params_values_comb[a])
        exec_res = self.simulator.sim('new_rp', self.inputs[self.iter],
                                      params_to_use, self.available_cpu)
        lat = np.sum(exec_res.get('latency'))
        util = self.simulator.get_utility(params_to_use)
        self.curr_state = self.encode_state(lat, a)
        finished = False
        self.iter += 1
        if self.iter >= len(self.inputs):
            finished = True
        return (self.curr_state, self.compute_reward(lat, util), finished, (lat, util, self.available_cpu))

    def reset(self):
        self.iter = 0
        self.curr_state = self.encode_state(0.9, 0)
        return self.curr_state