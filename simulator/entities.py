#===================================================================================================#
#                                       Simulator Entities                                          #
#    Last Modification: 12.03.2020                                         Mauricio Fadel Argerich  #
#===================================================================================================#

from abc import ABC, abstractmethod
import cloudpickle
import numpy as np
from collections import OrderedDict, defaultdict
import psutil
from abc import ABC, abstractmethod
import sys


# =================================== CLASSES ===================================#
def params_input_to_string(params_data, io):
    res = ''
    # Param values are sorted by param_name.
    for p, v in sorted(params_data.items()):
        res += str(p) + ':' + str(v) + '_'

    res += str(io.io_id)

    return res


class AdASIO:
    def __init__(self, io_value = None, io_id = None, io_size = None, io_format = None):
        '''
        Initiate an Input/Output of a function. It can be created using the actual IO or
        using the IO id, size and format. You can also create an IO defining all its properties.
        '''
        if io_value == None and (io_id == None or io_size == None or io_format == None):
            raise ValueError('IO can be created using the actual IO (io_value) OR by using all of its data (io_id, io_size and io_format).')
            
        # If any property was defined use it, otherwise we fill it automatically.
        self.io_value  = io_value
        self.io_id     = io_id     if io_id     else str(io_value)
        self.io_size   = io_size   if io_size   else sys.getsizeof(io_value)
        self.io_format = io_format if io_format else type(io_value)
        
        
class AdASCpu:
    cores = 1
    speed = 1.0

    def __init__(self, cores=1, speed=1.0):
        self.cores = cores
        self.speed = speed


class AdASDevice:
    device_id = ''
    cpu = None
    memory = 0

    def __init__(self, device_id, cpu=None, memory=None):
        self.device_id = device_id
        self.cpu = cpu
        self.memory = memory
        
        
class AdASExecutionData:
    input_data = None
    params_data = None
    output_data = None
    stats = None

    def __init__(self, input_data = None, params_data = None, output_data = None, stats = None):
        self.input_data = input_data
        self.params_data = params_data
        self.output_data = output_data
        self.stats = stats
        
    def get_name(self):
        return params_input_to_string(self.params_data, self.input_data)
        
    
class AdASFunction:

    def __init__(self, function, params, deployments = None):
        self.function = function
        self.data =  {}
        self.params = OrderedDict()
        for param_name in params.keys():
            self.params[param_name] = (OrderedDict(params.get(param_name)))
            
    def get_dummy_params(self, params):
        dummy_params = {}
        for p, values in params.items():
            dummy_values = {}
            for v, u in values.items():
                dummy_values[str(v)] = u
            dummy_params[p] = dummy_values
        return dummy_params
    
    def add_exec_sample(self, device_id, input_data, output_data, params_data, stats):
        if self.data.get(device_id) == None:
            self.data[device_id] = defaultdict(list)

        exec_sample = AdASExecutionData(input_data=input_data, output_data=output_data,
                                        params_data=params_data, stats=stats)
        
        self.data.get(device_id)[exec_sample.get_name()].append(exec_sample)
        
    def get_utility(self, param_values):
        u = 0
        for p, v in param_values.items():
            u += self.params.get(p).get(v)
        return u

    def sim(self, device_id, input_data, param_values, exact_match = True):
        exec_sample = AdASExecutionData(input_data=input_data, params_data=param_values)
        
        if self.data.get(device_id) == None or self.data.get(device_id).get(exec_sample.get_name()) == None:
            raise ValueError('There is no data for this input, device and params!', device_id, exec_sample.get_name())

        return np.random.choice(self.data.get(device_id).get(exec_sample.get_name()))


class AdASProfile:
    device = None
    pipeline_data = None
    inputs_data = None

    def __init__(self, device = None, pipeline = None, inputs = None):
        self.device = device
        self.pipeline_data = pipeline
        self.inputs_data = inputs


class AdASMetric(ABC):

    def __init__(self):
        self.value = None

    @property
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def measure_start(self):
        return None

    @abstractmethod
    def measure_end(self, start = None):
        return None

    @abstractmethod
    def get_state_value(self):
        raise NotImplementedError

    @abstractmethod
    def combine_measurements_for_pipeline(pipeline_measurements):
        raise NotImplementedError


class AdASMetricLatency(AdASMetric):

    def __init__(self):
        self.name = 'latency'
        self.value = 0.0

    def measure_start(self):
        return time.timer()

    def measure_end(self, start = None):
        return time.timer() - start

    def get_state_value(self, target):
        s = self.value/target
        if s < 0.50:
            return 0
        elif s < 0.80:
            return 1
        elif s < 1.0:
            return 2
        elif s < 1.2:
            return 3
        elif s < 1.5:
            return 4
        else:
            return 5

    def combine_measurements_for_pipeline(pipeline_measurements):
        return np.sum([pm.value for pm in pipeline_measurements])