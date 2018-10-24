#===================================================================================================#
#                                       Simulator Entities                                          #
#    Last Modification: 06.08.2018                                         Mauricio Fadel Argerich  #
#===================================================================================================#

from abc import ABC, abstractmethod
import cloudpickle
import numpy as np
from collections import OrderedDict
import psutil
from abc import ABC, abstractmethod


# =================================== CLASSES ===================================#
class MFAUtil:
    def params_input_to_string(params_data, input_data):
        res = ''
        # Param values are sorted by param_name.
        for p, v in sorted(params_data.items()):
            res += str(p) + ':' + str(v) + '_'

        res += input_data.io_id

        return res


class MFAIOData:
    io_id = ''
    io_size = 0
    io_format = ''

    def __init__(self, io_id, io_size, io_format):
        self.io_id = io_id
        self.io_size = io_size
        self.io_format = io_format


class MFAIO:
    io_value = None
    data = None

    def __init__(self, io_id, io_size, io_format, io_value):
        self.io_value = io_value
        self.data = MFAIOData(io_id, io_size, io_format)


class MFACpu:
    cores = 1
    speed = 1.0

    def __init__(self, cores=1, speed=1.0):
        self.cores = cores
        self.speed = speed


class MFADevice:
    device_id = ''
    cpu = None
    memory = 0

    def __init__(self, device_id, cpu=None, memory=None):
        self.device_id = device_id
        self.cpu = cpu
        self.memory = memory


class MFAExecutionData:
    input_data = None
    params_data = None
    output_data = None
    stats = None

    def __init__(self, input_data = None, params_data = None, output_data = None, stats = None):
        self.input_data = input_data
        self.params_data = params_data
        self.output_data = output_data
        self.stats = stats


class MFAFunctionData:
    name = ''
    params_data = []  # list of dicts = {value: utility}
    deployments = {}  # dict of {device: exec_history}

    def __init__(self, name, params, deployments = None):
        self.name = name
        self.params_data = self.get_dummy_params(params)
        if not deployments:
            self.deployments = {}
        else:
            self.deployments = deployments

    def get_dummy_params(self, params):
        dummy_params = {}
        for p, values in params.items():
            dummy_values = {}
            for v, u in values.items():
                dummy_values[str(v)] = u
            dummy_params[p] = dummy_values
        return dummy_params

    def add_exec_sample(self, device_id, input_data, output_data, params_data, stats):
        exec_history = self.deployments.get(device_id)
        if exec_history == None:
            exec_history = {}

        if exec_history.get(MFAUtil.params_input_to_string(params_data, input_data)) == None:
            exec_history[MFAUtil.params_input_to_string(params_data, input_data)] = []

        exec_sample = MFAExecutionData(input_data=input_data,
                                       output_data=output_data,
                                       params_data=params_data,
                                       stats=stats)
        exec_history.get(MFAUtil.params_input_to_string(params_data, input_data)).append(exec_sample)

        self.deployments[device_id] = exec_history

    def get_utility(self, param_values):
        u = 0
        for p, v in param_values.items():
            u += self.params_data.get(p).get(v)
        return u

    def sim(self, device_id, input_data, param_values, exact_match = True):
        exec_history = self.deployments.get(device_id)
        if exec_history == None or exec_history.get(MFAUtil.params_input_to_string(param_values, input_data)) == None:
            raise ValueError('There is no data for this input, device and these params!',
                             device_id,
                             MFAUtil.params_input_to_string(param_values, input_data))

        exec_sample = np.random.choice(exec_history.get(MFAUtil.params_input_to_string(param_values, input_data)))

        return (exec_sample.output_data, exec_sample.stats, self.get_utility(param_values))


class MFAFunction:
    function = None
    params = OrderedDict()
    data = None

    def __init__(self, function, params, deployments = None):
        self.function = function
        self.data = MFAFunctionData(function.__name__, params, deployments)
        self.params = OrderedDict()
        for param_name in params.keys():
            self.params[param_name] = (OrderedDict(params.get(param_name)))


class MFAProfileData:
    device = None
    pipeline_data = None
    inputs_data = None

    def __init__(self, device = None, pipeline = None, inputs = None):
        self.device = device
        self.pipeline_data = pipeline
        self.inputs_data = inputs


class MFAMetric(ABC):

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


class MFAMetricLatency(MFAMetric):

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