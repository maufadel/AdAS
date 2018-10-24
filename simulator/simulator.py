#===================================================================================================#
#                                         Simulator                                                 #
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


class MFASimulator:

    def __init__(self):
        self.profile = None
        self.utility_map = pd.read_csv('scripts/simulator/comb_summary.csv', index_col = 0)

    def check_pipelines(self, pipeline1, pipeline2):
        if len(pipeline1) != len(pipeline2):
            return False
        else:
            for i in range(len(pipeline1)):
                if pipeline1[i].name != pipeline2[i].name:
                    return False

        return True

    def add_deployments(self, function, deployments):
        for d in deployments.keys():
            if function.deployments.get(d) == None:
                # Data is from different devices.
                function.deployments[d] = deployments.get(d)
            else:
                # Same device
                for k in deployments.get(d).keys():
                    if function.deployments.get(d).get(k) == None:
                        function.deployments.get(d)[k] = []
                    function.deployments.get(d).get(k).extend(deployments.get(d).get(k))

    def add_profile(self, profile_to_load):
        if self.check_pipelines(self.profile.pipeline_data, profile_to_load.pipeline_data):
            for i in range(len(self.profile.pipeline_data)):
                self.add_deployments(self.profile.pipeline_data[i], profile_to_load.pipeline_data[i].deployments)

    def load_profile(self, filename):
        profile_to_load = None
        with open(filename, "rb") as f:
            profile_to_load = cloudpickle.load(f)

        if self.profile == None:
            self.profile = profile_to_load
        else:
            self.add_profile(profile_to_load)

    def get_utility(self, param_values):
        '''
        u = 0
        for f in self.profile.pipeline_data:
            f_params = {}
            for p, v in param_values.items():
                if f.params_data.get(p):
                    f_params[p] = v

            u += f.get_utility(f_params)
        '''
        
        comb_id = param_values.get('scale_factor') + ',' + param_values.get('face_cascade') + ',' + param_values.get('resize') + ',' + param_values.get('face_recognizer')

        return self.utility_map.loc[comb_id, 'target_precision']

    def sim(self, device_id, input_data, param_values, available_cpu = 1.0):
        res_stats = defaultdict(list)
        f_output = input_data
        for f in self.profile.pipeline_data:
            f_params = {}
            for p, v in param_values.items():
                if f.params_data.get(p):
                    f_params[p] = v

            f_output, stats, utility = f.sim(device_id, f_output, f_params)
            for k, v in stats.items():
                res_stats[k].append(v / available_cpu)

            res_stats['utility'].append(utility)

        return res_stats