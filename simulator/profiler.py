#===================================================================================================#
#                                    Pipeline-Device Profiler                                       #
#    Last Modification: 13.03.2020                                         Mauricio Fadel Argerich  #
#===================================================================================================#

import cloudpickle
import itertools
import pandas as pd
import psutil
import sys
import time
from tqdm import tqdm

import entities
from entities import AdASFunction, AdASDevice, AdASProfile, AdASCpu, AdASIO


def save(filename, device, pipeline, inputs):
    profile = AdASProfile(device, pipeline, inputs)

    with open(filename, "wb") as f:
        cloudpickle.dump(profile, f)


def get_device(name):
    '''
    Create a device with current device configuration and return it.
    '''
    return AdASDevice(name,
                      cpu=AdASCpu(cores=psutil.cpu_count(), speed=2.0), #psutil.cpu_freq().current),
                      memory=psutil.virtual_memory().total)


def exec_combination(device, pipeline, pi_dict, f_input, combination):
    '''
    Execute the whole pipeline for the given input and combination of parameters.
    Note that the order of the parameter values must be the same as in the ordered
    dictionary function.params.keys().
    '''
    row_dict = pi_dict.copy()
    base_f_c = 0
    for f in pipeline:
        params_list = list(f.params.keys())
        params_dict = {}
        for param_i in range(len(params_list)):
            p_value = combination[base_f_c + param_i]
            params_dict[params_list[param_i]] = p_value
            row_dict[f.function.__name__  + '_' + str(params_list[param_i])] = str(p_value)

        base_f_c += len(f.params.keys())
        row_dict['cpu_idle'] = psutil.cpu_times_percent(interval=0.5).idle
        row_dict['memory_available'] = psutil.virtual_memory().available
        row_dict[f.function.__name__ + '_start'] = time.time()
        row_dict[f.function.__name__ + '_end'] = time.time()
        f_output_value = f.function(f_input.io_value, **params_dict)
        f_output = AdASIO(io_id='io_' + entities.params_input_to_string(params_dict, f_input),
                          io_value=f_output_value)

        f.add_exec_sample(f_input, f_output, params_dict,
                          {'latency': (row_dict[f.function.__name__ + '_end'] - row_dict[f.function.__name__ + '_start']),
                           'utility': f.get_utility(params_dict)})
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
    res_cols.append('io_value')
    res_cols.append('io_id')
    res_cols.append('io_size')
    res_cols.append('io_format')

    # Create params_list to create combinations of param values and
    # add function and params to columns.
    params_list = []
    for f in pipeline:
        res_cols.append(f.function.__name__ + '_start')
        res_cols.append(f.function.__name__ + '_end')
        for param_name in f.params.keys():
            res_cols.append(f.function.__name__ + '_' + param_name)
            params_list.append(list(f.params.get(param_name).keys()))

    all_combinations = list(itertools.product(*params_list))

    # Add system status columns.
    res_cols.append('cpu_idle')
    res_cols.append('memory_available')

    # We will create a dictionary for each row of the results and save the results when the loop 
    # for every input is finished.
    res_rows = []

    for pi in pipeline_inputs:
        print(pi.io_id, flush=True)
        pi_dict = {}
        # Input data is saved in this dictionary that will be part
        # of the results row.
        pi_dict['io_value'] = str(pi.io_value)
        pi_dict['io_id'] = str(pi.io_id)
        pi_dict['io_size'] = str(pi.io_size)
        pi_dict['io_format'] = str(pi.io_format)

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