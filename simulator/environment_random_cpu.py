#===================================================================================================#
#                                      AdAS Environment                                             #
#    Last Modification: 09.03.2020                                         Mauricio Fadel Argerich  #
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

    def __init__(self, simulator, latency_target, inputs, cpu_steps):
        # cpu_steps is not used.
        self.simulator = simulator
        self.latency_target = latency_target
        self.inputs = inputs
        self.available_cpu = self.get_new_available_cpu(0.3 + abs(np.random.randn())*0.7)

        self.set_combinations(simulator.profile.pipeline_data)

        # CPU, latency and one discrete value sorted by ascending utility for each param.
        low = np.concatenate((np.array([0, 0]), np.zeros(len(self.params_names))))
        high = np.concatenate(([100, np.Infinity], 
                               [len(self.params_names_values.get(pn)) for pn in self.params_names]))
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        self.action_space = spaces.Discrete(len(self.params_values_comb))

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

        
    def step(self, a):
        # action is 0 to decrease, 1 to stay, 2 to increase
        # map the number to the actual combination
        # simulate the inputs with the combination
        # get latency and utility
        # return (state, reward, finished?, {info})
        self.available_cpu = self.get_new_available_cpu(self.available_cpu)
        params_to_use = self.get_params_values(self.params_values_comb[a])
        exec_res = self.simulator.sim('new_rp', self.inputs[self.iter],
                                      params_to_use, self.available_cpu)
        self.curr_lat = np.sum(exec_res.get('latency')) # Sum of latencies of each function.
        self.curr_util = self.simulator.get_utility(params_to_use)
        
        # CPU, latency and one discrete value for each param.
        self.curr_state = np.concatenate(([self.available_cpu, self.curr_lat],
                                          [self.get_value_ix(pn, params_to_use.get(pn)) for pn in self.params_names]))
        
        finished = False
        self.iter += 1
        if self.iter >= len(self.inputs):
            finished = True
            
        # Create info dictionary.
        info = OrderedDict()
        info['available_cpu'] = self.available_cpu
        info['latency'] = self.curr_lat
        for pn, v in self.get_params_values(self.params_values_comb[a]).items():
            info[pn] = str(self.get_value_ix(pn, v)) + ': ' + str(v)
        info['utility'] = self.curr_util
        info['step'] = self.iter
        
        return self.curr_state, self.compute_reward(self.curr_lat, self.curr_util), finished, info

    def reset(self):
        self.iter = 0
        self.available_cpu = 1.0
        self.curr_lat = 0.9
        self.curr_state = np.concatenate(([self.available_cpu, self.curr_lat],
                                          [0 for _ in self.params_names]))
        return self.curr_state
    
class MFAEnvironmentSimpleStates:

    def __init__(self, simulator, latency_target, inputs, cpu_steps):
        # cpu_steps is not used.
        self.simulator = simulator
        self.latency_target = latency_target
        self.inputs = inputs
        self.available_cpu = self.get_new_available_cpu(0.3 + abs(np.random.randn())*0.7)

        self.set_combinations(simulator.profile.pipeline_data)
        self.state_map = self.build_state_map()
        self.observation_space = len(self.state_map.keys())   # latency * comb_params
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
        return self.state_map.get((lat_state, params_comb))

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