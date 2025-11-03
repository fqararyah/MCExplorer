#Question1: six hump camelback function.
import random
import math
import mapping_utils.custom_mapping_utils as custom_mapping_utils
from mapping_types.mapping_description import *
from preformance_record import *
from mapping_types.generic_mapping import *
import copy
import os
import time

x_min_max = [-3, 3] #the function is defined between: -3<x<3
y_min_max = [-2, 2] #the function is defined between: -2<y<2
stopping_if_no_acceptance = 25 #stopping criteria, option1: after 10 itearions without accepting new solution.
inner_iterations = 1000 #iterations of the inner loop
neighborhood_range = 1
stopping_temperature = 0.001
t_N = 0.001
alpha = 0.95

def obj_f(mapping, metric): #the objective function of six hump camelback 
    if metric == Metrics.LATENCY:
        return mapping.calc_exec_time()
    elif metric == Metrics.THROUGHPUT:
        return mapping.calc_throughput()
    elif metric == Metrics.BUFFER:
        return mapping.calc_on_chip_buffer_sz()
    elif metric == Metrics.ACCESS:
        return mapping.calc_off_chip_weights_access() + mapping.calc_off_chip_fms_access()
    elif metric == Metrics.ENERGY:
        return mapping.calc_energy()
        

if not constants.TEST_SEEDS:
    random.seed(5) #setting the seed, will be changed 10 times while experimentig.
x = 0 #arbitrary start, will be changed 5 times while experimentig.
y = 0 #arbitrary start, will be changed 5 times while experimentig.

# select t_0 that is large enough to heat the system,
# at the same time not exteremly large to avoid spending a lot
# of time to cool down.
def pick_t_0(number_of_steps, mapping, mapping_desc, metric, ideal_value = -1):
    energy_diffs = []
    last_energy = ideal_value 
    for i in range(number_of_steps):
        current_energy = obj_f(mapping, metric)
        mapping, mapping_desc = pick_a_neighbor(mapping_desc)
        energy_diffs.append(abs(current_energy - last_energy))
        last_energy = current_energy
        if i == 0 and ideal_value == -1:
            energy_diffs[0] = 0
    
    return max(energy_diffs)

#switch between two stopping criteria easily.
def stop_now(criteria_type, num_iterations, no_acceptance = -1, iterations = -1, temperature = math.inf):
    if criteria_type == 'no_acceptance':
        return no_acceptance > stopping_if_no_acceptance
    elif criteria_type == 'iterations':
        return iterations >= num_iterations
    elif criteria_type == 'temperature':
        return temperature < stopping_temperature

#function to pick a neighbor, guarantees staying in the boundaries:
def pick_a_neighbor(mapping_desc):
    return custom_mapping_utils.modeify_mapping_random(mapping_desc)

def cool_schedule_1(t_0, t_N, i, N):
    return t_0 - i * (t_0 - t_N) / N

def cool_schedule_2(t_0, t_N, i, N):
    return t_0 * math.pow(t_N / t_0, i / N)

def cool_schedule_3(t_0, t_N, i, N):
    return (t_0 - t_N) / (1 + math.exp( .3 * (i - N/2) ) ) + t_N

def exp_cooling(t_0, alpha, iter):
    return t_0 * (alpha ** iter)

def run_simulated_annealing(mapping, mapping_desc, metric, num_iterations):
    t_0 = pick_t_0(100, mapping, mapping_desc, metric)# random walk of size 100 to pick good enough t_0
    t = t_0
    obj = obj_f(mapping, metric)
    best_so_far_mapping_desc = mapping_desc
    current_iteration = 0
    iterations_without_acceptance = 0 # note in this code sample it is modified but not used as stopping conditions, it was used in experimenting.
    best_so_far = obj
    print('run_simulated_annealing for {} iteration ...'.format(num_iterations))
    num_iterations = num_iterations // inner_iterations
    start_time = time.time()
    while t != 0 and not stop_now(criteria_type = 'iterations', num_iterations=num_iterations, iterations=current_iteration):
        for i in range(0, inner_iterations):
            accept = False
            if i == 0:
                new_mapping, new_mapping_desc =\
                    custom_mapping_utils.generate_random_mapping(
                        mapping_desc.board_name, mapping_desc.model_name, mapping_desc.model_dag, 
                        num_segments= -1, timing_metric= mapping_desc.timing_metric)
            else:
                new_mapping, new_mapping_desc = pick_a_neighbor(mapping_desc)

            new_obj = obj_f(new_mapping, metric)
            delta = abs(new_obj - obj)
            if first_better_than_second(metric, new_obj, obj):
                accept = True
            else:
                rand = random.random()
                if rand < math.exp(-delta / t):
                    accept = True
            if accept:   
                mapping_desc = new_mapping_desc
                obj = new_obj
                if first_better_than_second(metric, obj, best_so_far):
                    best_so_far = obj # 
                    best_so_far_mapping_desc = copy.deepcopy(mapping_desc)
                iterations_without_acceptance = 0 # 
            else:
                iterations_without_acceptance += 1
            
        current_iteration += 1
        t = exp_cooling(t_0, alpha, current_iteration) #cool_schedule_2(t_0, t_N, current_iteration, num_iterations)

    duration = time.time() - start_time

    timing_file_name = 'sa_iterations_{}.json'.format(num_iterations)
    timings_dict = mapping_general_utils.load_json_to_dict(
        os.path.dirname(__file__) + '/timing/' + timing_file_name)

    if timings_dict != None:
        new_timing = duration
        if mapping_desc.board_name in timings_dict:
            if mapping_desc.model_name in timings_dict[mapping_desc.board_name]:
                previous_runs = int(
                    timings_dict[mapping_desc.board_name][mapping_desc.model_name]['runs'])
                previous_timing = float(
                    timings_dict[mapping_desc.board_name][mapping_desc.model_name]['duration'])
                previous_timing *= previous_runs
                new_timing = (previous_timing + duration) / (1 + previous_runs)
            else:
                timings_dict[mapping_desc.board_name][mapping_desc.model_name] = {}
        else:
            timings_dict[mapping_desc.board_name] = {mapping_desc.model_name: {}}
        if 'runs' not in timings_dict[mapping_desc.board_name][mapping_desc.model_name]:
            timings_dict[mapping_desc.board_name][mapping_desc.model_name]['runs'] = 0
        timings_dict[mapping_desc.board_name][mapping_desc.model_name]['runs'] += 1
        timings_dict[mapping_desc.board_name][mapping_desc.model_name]['duration'] = new_timing
    else:
        timings_dict = {}
        timings_dict[mapping_desc.board_name] = {mapping_desc.model_name: {}}
        timings_dict[mapping_desc.board_name][mapping_desc.model_name]['runs'] = 1
        timings_dict[mapping_desc.board_name][mapping_desc.model_name]['duration'] = duration

    mapping_general_utils.save_dict_to_json(timings_dict, os.path.dirname(__file__) + '/timing/',
                                            timing_file_name)
     
    return best_so_far, best_so_far_mapping_desc
