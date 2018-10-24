#===================================================================================================#
#                                    Pipeline-Device Profiler                                       #
#    Last Modification: 08.08.2018                                         Mauricio Fadel Argerich  #
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

sys.path.append('scripts/simulator/')
sys.path.append('scripts/utils/')

from entities import MFADevice, MFACpu
from simulator import MFASimulator


class MFAEnvironment:

    def __init__(self, simulator, latency_target, inputs, cpu_steps, steps = 1000):
        # cpu_steps is not used.
        self.simulator = simulator
        self.latency_target = latency_target
        self.available_cpu = self.get_new_available_cpu(0.3 + abs(np.random.randn())*0.7)
        self.inputs = inputs
        self.curr_input = np.random.choice(self.inputs)
        self.steps = steps

        self.set_combinations(simulator.profile.pipeline_data)
        self.state_map = self.build_state_map()
        self.observation_space = len(self.state_map.keys())   # CPU_states * latency * comb_params
        self.action_space = len(self.params_values_comb) # up, stay and down
        self.reset()
        
    def get_new_input(self, old_input):
        if abs(np.random.randn()) < 0.1:
            new_input = np.random.choice(self.inputs)
        else:
            new_input = old_input
            
        return new_input
        
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
        for c in list(itertools.product(range(3), range(3), range(len(self.params_values_comb)))):
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

    def discretize_cpu(self, cpu_available):
        if cpu_available < 0.5:
            return 0
        elif cpu_available < 0.85:
            return 1
        else:
            return 2

    def encode_state(self, lat, cpu, params_comb):
        lat_state = self.discretize_lat(lat)
        cpu_state = self.discretize_cpu(cpu)
        return self.state_map.get((lat_state, cpu_state, params_comb))

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
        util = self.simulator.get_utility(params_to_use) #np.sum(exec_res.get('utility'))
        self.curr_state = self.encode_state(lat, self.available_cpu, a)
        finished = False
        self.iter += 1
        if self.iter >= self.steps:
            finished = True
        return (self.curr_state, self.compute_reward(lat, util), finished, (lat, util))

    def reset(self):
        self.iter = 0
        self.curr_state = self.encode_state(0.9, 1.0, 0)
        return self.curr_state
    
class MFAEnvironmentSimpleStates:

    def __init__(self, simulator, latency_target, inputs, cpu_steps, steps = 1000):
        # cpu_steps is not used.
        self.simulator = simulator
        self.latency_target = latency_target
        self.available_cpu = self.get_new_available_cpu(0.3 + abs(np.random.randn())*0.7)
        self.inputs = inputs
        self.curr_input = np.random.choice(self.inputs)
        self.steps = steps

        self.set_combinations(simulator.profile.pipeline_data)
        self.state_map = self.build_state_map()
        self.observation_space = len(self.state_map.keys())   # CPU_states * latency * comb_params
        self.action_space = len(self.params_values_comb) # up, stay and down
        self.reset()
        
    def get_new_input(self, old_input):
        if abs(np.random.randn()) < 0.1:
            new_input = np.random.choice(self.inputs)
        else:
            new_input = old_input
            
        return new_input
        
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
        if self.iter >= self.steps:
            finished = True
        return (self.curr_state, self.compute_reward(lat, util), finished, (lat, util))

    def reset(self):
        self.iter = 0
        self.curr_state = self.encode_state(0.9, 0)
        return self.curr_state