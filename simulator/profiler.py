#===================================================================================================#
#                                    Pipeline-Device Profiler                                       #
#    Last Modification: 30.07.2018                                         Mauricio Fadel Argerich  #
#===================================================================================================#

import cloudpickle
import itertools
import pandas as pd
import psutil
import sys
import time
from tqdm import tqdm

from entities import MFAFunction, MFADevice, MFAProfileData, MFACpu, MFAIOData, MFAUtil, MFAIO


def save(filename, device, pipeline, inputs):
    pipeline_data = [f.data for f in pipeline]
    inputs_data = [i.data for i in inputs]
    profile_data = MFAProfileData(device, pipeline_data, inputs_data)

    with open(filename, "wb") as f:
        cloudpickle.dump(profile_data, f)


def get_device(name):
    '''
    Create a device with current device configuration and return it.
    '''
    return MFADevice(name,
                    cpu=MFACpu(cores=psutil.cpu_count(), speed=2.0), #psutil.cpu_freq().current),
                    memory=psutil.virtual_memory().total)


def exec_combination(device, pipeline, pi_dict, f_input, combination):
    '''
    Execute the whole pipeline for the given function and combination of parameters.
    Note that the order of the parameter values must be the same as in the ordered
    dictionary function.params.keys().
    '''
    row_dict = pi_dict.copy()
    base_f_c = 0
    for f in pipeline:
        params_list = list(f.params.keys())
        params_dict = {}
        params_data = {}
        for param_i in range(len(params_list)):
            p_value = combination[base_f_c + param_i]
            params_dict[params_list[param_i]] = p_value
            row_dict[f.data.name + '_' + str(params_list[param_i])] = str(p_value)
            params_data[str(params_list[param_i])] = str(p_value)

        row_dict['cpu_idle'] = psutil.cpu_times_percent(interval=0.5).idle
        row_dict['memory_available'] = psutil.virtual_memory().available
        row_dict[f.data.name + '_start'] = time.time()
        f_output_value = f.function(f_input.io_value, **params_dict)
        row_dict[f.data.name + '_end'] = time.time()
        base_f_c += len(f.params.keys())

        f_output = MFAIO(io_id='io_' + MFAUtil.params_input_to_string(params_dict, f_input.data),
                                io_size=sys.getsizeof(f_output_value),
                                io_format=type(f_output_value),
                                io_value=f_output_value)

        f.data.add_exec_sample(device.device_id,
                                f_input.data,
                                f_output.data,
                                params_data,
                                {'latency': (row_dict[f.data.name + '_end'] - row_dict[f.data.name + '_start'])})

        f_input = f_output

    row_dict['output'] = f_input.io_value

    return row_dict


def profile(device_name, pipeline, pipeline_inputs, n, results_filename = 'results.csv'):
    '''
    Run the whole pipeline n times for the given pipeline_inputs,
    for each different combination of parameter values of the pipeline
    functions. Results will be saved to results_filename in csv format.
    '''
    # Save current device information.
    device = get_device(device_name)
    save('profile_' + device_name + '.pkl', device, pipeline, pipeline_inputs)

    # Create columns for results dataframe.
    res_cols = []

    # Data from inputs.
    for k in pipeline_inputs[0].data.__dict__.keys():
        res_cols.append(k)

    # Create params_list to create combinations of param values and
    # add function and params to columns.
    params_list = []
    for f in pipeline:
        res_cols.append(f.data.name + '_start')
        res_cols.append(f.data.name + '_end')
        for param_name in f.params.keys():
            res_cols.append(f.data.name + '_' + param_name)
            params_list.append(list(f.params.get(param_name).keys()))

    all_combinations = list(itertools.product(*params_list))

    # Add system status columns.
    res_cols.append('cpu_idle')
    res_cols.append('memory_available')

    # We will create a dictionary for each row of the results
    # and save the results when the loop for every input is
    # finished.
    res_rows = []

    for pi in pipeline_inputs:
        print(pi.data.io_id, flush=True)
        pi_dict = {}
        # Input data is saved in this dictionary that will be part
        # of the results row.
        for k, v in pi.data.__dict__.items():
            pi_dict[k] = v

        for comb in tqdm(all_combinations.copy()):
            try:
                for _ in range(n):
                    res_rows.append(exec_combination(device, pipeline, pi_dict, pi, comb))
            except:
                # Remove combination so we don't try to run it again.
                all_combinations.remove(comb)
                print()
                print("Combination failed!", end=' ')
                for pv in comb:
                    print(pv, end=' ')
                print()

        # Save results every time we finish all of the combination for an input.
        res = pd.DataFrame(res_rows)
        res.to_csv(results_filename)
        save('profile_' + device_name + '.pkl', device, pipeline, pipeline_inputs)
        print()