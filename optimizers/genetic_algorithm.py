import random
import time
import mapping_utils.custom_mapping_utils as custom_mapping_utils
from mapping_types.mapping_description import *
from preformance_record import *
from mapping_types.generic_mapping import *
import copy
from tqdm import tqdm
import os

if not constants.TEST_SEEDS:
    random.seed(5)

crossover_probability = 0.9
mutation_probability = 0.1
elite_percentage = 0.01
mutable_elite_percentage = 0.01
tournament_size = 2
min_population_size = 40


def generate_population(population_size, board_name, model_name, model_dag, min_segments, max_engines, timing_metric):
    population = []
    ce_possibilities = max_engines - min_segments + 1
    
    population_per_ce_count = (
        population_size + ce_possibilities - 1) // ce_possibilities

    for num_segments in range(min_segments, max_engines + 1):
        for j in range(population_per_ce_count):
            if len(population) >= population_size:
                break
            _, mapping_desc =\
                custom_mapping_utils.generate_random_mapping(
                    board_name, model_name, model_dag, num_segments, timing_metric)
            population.append(mapping_desc)

    return population


def get_elite_count_from_percentage(percentage, population_size):
    elite_count = int(percentage * population_size)
    if percentage > 0 and elite_count % 2 != 0:
        elite_count += 1
    if percentage != 0 and elite_count == 0:
        elite_count = 2

    return elite_count


def member_fitness(mapping, metric):  # the objective function of six hump camelback
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


def get_fitness_values(population, population_size, metric):
    fitness_val_list = []
    for i in range(population_size):
        mapping_desc = population[i]
        mapping_desc_dict = custom_mapping_utils.prepare_custom_mapping_desc(
            mapping_desc.segment_layers_list, mapping_desc.segment_block_list, mapping_desc.block_engines_list)
        
        mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
            mapping_desc.board_name, mapping_desc.model_dag, mapping_desc_dict, timing_metric=mapping_desc.timing_metric,
            adjust_pes=mapping_desc.adjust_pes)
        fitness_val_list.append(member_fitness(mapping, metric))

    return fitness_val_list


def sort_population_by_fitness(fitness_val_list, population, population_size, metric):
    sorted_population = []

    indices_list = [*range(len(fitness_val_list))]
    zipped_lists = zip(fitness_val_list, indices_list)
    zipped_lists = sorted(
        zipped_lists, reverse=(metric == Metrics.THROUGHPUT))
    _, indices_list = zip(*zipped_lists)
    for i in range(population_size):
        sorted_population.append(population[indices_list[i]])

    return sorted_population


def sort_population_by_segment_count(population):
    population_engine_map = {}
    for member in population:
        if member.num_segments not in population_engine_map:
            population_engine_map[member.num_segments] = []
        population_engine_map[member.num_segments].append(member)

    i = 0
    for num_engines, members in population_engine_map.items():
        for member in members:
            population[i] = member
            i += 1

    return population


def get_fittest_parents_roulette_wheel(fitness_val_list, population, population_size, metric):
    parent_list = []
    non_elite_parent_list = []
    sum_fitnesses = sum(fitness_val_list)
    wheel_chances = []
    elite_count = get_elite_count_from_percentage(
        elite_percentage, population_size)
    mutable_elite_count = get_elite_count_from_percentage(
        mutable_elite_percentage, population_size)

    if elite_percentage > 0:
        sorted_population = sort_population_by_fitness(
            fitness_val_list, population, population_size, metric)

    for i in range(elite_count):
        parent_list.append(sorted_population[i])
    for i in range(mutable_elite_count):
        parent_list.append(sorted_population[i])

    for i in range(0, len(fitness_val_list)):
        wheel_chances.append(fitness_val_list[i]/sum_fitnesses)

    selected = elite_count + mutable_elite_count
    min_wheel_chance = min(wheel_chances) if metric == Metrics.LATENCY else 0
    max_wheel_chance = max(wheel_chances) if metric != Metrics.LATENCY else 1

    # repitition is allowed
    while selected < population_size:
        a_prob = random.uniform(min_wheel_chance, max_wheel_chance)
        starting_point = random.randint(0, population_size)
        for i in range(starting_point, population_size):
            if first_better_than_or_equal_second(metric, wheel_chances[i], a_prob):
                non_elite_parent_list.append(population[i])
                selected += 1
                # if selected == population_size:
                break
    sort_population_by_segment_count(non_elite_parent_list)

    parent_list.extend(non_elite_parent_list)

    return parent_list



def get_fittest_parents_tournament(population, population_size, metric):
    parent_list = []
    non_elite_parent_list = []
    elite_count = get_elite_count_from_percentage(
        elite_percentage, population_size)
    mutable_elite_count = get_elite_count_from_percentage(
        mutable_elite_percentage, population_size)
    
    random.shuffle(population)
    fitness_val_list = get_fitness_values(population, population_size, metric)
    for i in range(0, population_size - (elite_count + mutable_elite_count)):
        best_index = -1
        best_val = -1
        for j in range(0, tournament_size):
            rand_index = random.randint(0, population_size - 1)
            fitness_val = fitness_val_list[rand_index]
            if best_val == -1 or first_better_than_or_equal_second(metric, fitness_val, best_val):
                best_val = fitness_val
                best_index = rand_index
                
        non_elite_parent_list.append(population[best_index])

    if elite_percentage > 0:
        sorted_population = sort_population_by_fitness(
            fitness_val_list, population, population_size, metric)
    
    for i in range(elite_count):
        parent_list.append(sorted_population[i])
    for i in range(mutable_elite_count):
        parent_list.append(sorted_population[i])

    sort_population_by_segment_count(non_elite_parent_list)

    parent_list.extend(non_elite_parent_list)

    return parent_list

def get_fittest_parents_tournament_combined(population, fitness_val_list, population_size, metric):
    parent_list = []
    non_elite_parent_list = []
    elite_count = get_elite_count_from_percentage(
        elite_percentage, population_size)
    mutable_elite_count = get_elite_count_from_percentage(
        mutable_elite_percentage, population_size)

    tmp = list(zip(population, fitness_val_list))
    random.shuffle(tmp)
    population, fitness_val_list = zip(*tmp)
    for i in range(0, population_size - (elite_count + mutable_elite_count)):
        best_index = -1
        best_val = -1
        for j in range(0, tournament_size):
            rand_index = random.randint(0, 2 * population_size - 1)
            fitness_val = fitness_val_list[rand_index]
            if best_val == -1 or first_better_than_or_equal_second(metric, fitness_val, best_val):
                best_val = fitness_val
                best_index = rand_index

        non_elite_parent_list.append(population[best_index])

    if elite_percentage > 0:
        sorted_population = sort_population_by_fitness(
            fitness_val_list, population, 2 * population_size, metric)

    for i in range(elite_count):
        parent_list.append(sorted_population[i])
    for i in range(mutable_elite_count):
        parent_list.append(sorted_population[i])

    sort_population_by_segment_count(non_elite_parent_list)

    parent_list.extend(non_elite_parent_list)

    return parent_list


def do_crossover(parent_list, population_size):
    offspring_list = []

    elite_count = get_elite_count_from_percentage(
        elite_percentage, population_size)
    for i in range(elite_count):
        offspring_list.append(parent_list[i].copy_mapping_desc())

    itr = elite_count
    while (itr < population_size - 1):
        parent1_index = itr
        parent2_index = (itr+1)
        parent_1 = parent_list[parent1_index].copy_mapping_desc()
        parent_2 = parent_list[parent2_index].copy_mapping_desc()

        x = random.random()
        if x > crossover_probability or parent_1.num_segments != parent_2.num_segments \
                or parent_1.num_segments == 1:
            itr += 2
            offspring_list.append(parent_1)
            offspring_list.append(parent_2)
            continue

        num_segments = parent_1.num_segments
        if num_segments == 2:
            crossover_point = 1
        else:
            crossover_point = random.randint(1, num_segments - 2)
            
        _, offspring1_mapping_desc = custom_mapping_utils.mapping_from_crossover_of_two_mappings(
            parent_1, parent_2, crossover_point)
        offspring_list.append(offspring1_mapping_desc)
        _, offspring2_mapping_desc = custom_mapping_utils.mapping_from_crossover_of_two_mappings(
            parent_2, parent_1, crossover_point)
        offspring_list.append(offspring2_mapping_desc)

        itr += 2

    return offspring_list


def do_mutation(offspring_list, population_size):
    elite_count = get_elite_count_from_percentage(elite_percentage, population_size)
    for i in range(elite_count, population_size):
        random_value = random.random()
        if random_value > mutation_probability:
            continue
        _, offspring_list[i] = custom_mapping_utils.modeify_mapping_random(
            offspring_list[i])

    return offspring_list


def get_max_fitness(fitness_list, metric):
    if metric == Metrics.THROUGHPUT:
        return max(fitness_list)
    else:
        return min(fitness_list)


def adaptive_shrinking(population, fittness_values_list, max_fitness, metric, population_size):
    new_population = []
    if population_size == min_population_size:
        return population, population_size

    for i in range(population_size):
        if metric == Metrics.THROUGHPUT:
            if fittness_values_list[i] > 0.33 * max_fitness:
                new_population.append(population[i])
        else:
            if fittness_values_list[i] < 3 * max_fitness:
                new_population.append(population[i])

    new_size = len(new_population)
    for i in range(new_size, min_population_size):
        new_population.append(population[i - new_size])

    if len(new_population) % 2 != 0:
        new_population.append(new_population[0])

    return new_population, len(new_population)


def optimize(population, metric, population_size, number_of_generations):
    fitness_history = {'average': [], 'best': []}
    fittest_gene = population[fitness_val_list.index(max_fitness)]
    max_fitness = get_max_fitness(fitness_val_list, metric)
    fittest_gene = population[fitness_val_list.index(max_fitness)]
    average_fitness = sum(fitness_val_list) / population_size
    #print(average_fitness, max_fitness)
    fitness_history['average'].append(average_fitness)
    fitness_history['best'].append(max_fitness)
    for i in tqdm(range(number_of_generations)):
    # for i in range(number_of_generations):
        population = get_fittest_parents_tournament(population, population_size, metric)
        #population = get_fittest_parents_roulette_wheel(fitness, population, population_size, metric)
        offsprings = do_crossover(population, population_size)
        population = do_mutation(offsprings, population_size)
        fitness_val_list = get_fitness_values(population, population_size, metric)
        current_max_fitness = get_max_fitness(fitness_val_list, metric)
        average_fitness = sum(fitness_val_list) / population_size
        #print(average_fitness, max_fitness)
        if first_better_than_second(metric, current_max_fitness, max_fitness):
            max_fitness = current_max_fitness
            fittest_gene = population[fitness_val_list.index(max_fitness)].copy_mapping_desc()

        fitness_history['average'].append(average_fitness)
        fitness_history['best'].append(max_fitness)

        # population, population_size = adaptive_shrinking(
        #    population, fitness, max_fitness, metric, population_size)

    return max_fitness, fittest_gene, fitness_history

def optimize_v2(population, metric, population_size, number_of_generations):
    fitness_history = {'average': [], 'best': []}
    fitness_val_list = get_fitness_values(population, population_size, metric)
    max_fitness = get_max_fitness(fitness_val_list, metric)
    fittest_gene = population[fitness_val_list.index(max_fitness)].copy_mapping_desc()
    average_fitness = sum(fitness_val_list) / population_size
    fitness_history['average'].append(average_fitness)
    fitness_history['best'].append(max_fitness)
    for i in tqdm(range(number_of_generations)):
        # for i in range(number_of_generations):
        # for i in range(population_size):
        #     print('+', i, population[i])
        offsprings = do_crossover(population, population_size)
        # for i in range(population_size):
        #     print('++', i, population[i])
        do_mutation(offsprings, population_size)
        # for i in range(population_size):
        #     print('+++', i, population[i])
        combined_population = population + offsprings
        fitness_val_list = get_fitness_values(
            combined_population, population_size * 2, metric)
        population = get_fittest_parents_tournament_combined(combined_population, fitness_val_list,
                                                    population_size, metric)
        fitness_val_list = get_fitness_values(population, population_size, metric)
        current_max_fitness = get_max_fitness(fitness_val_list, metric)
        average_fitness = sum(fitness_val_list) / population_size

        if first_better_than_second(metric, current_max_fitness, max_fitness):
            max_fitness = current_max_fitness
            fittest_gene = population[fitness_val_list.index(max_fitness)].copy_mapping_desc()
        
        fitness_history['average'].append(average_fitness)
        fitness_history['best'].append(max_fitness)

        # population, population_size = adaptive_shrinking(
        #    population, fitness, max_fitness, metric, population_size)

    return max_fitness, fittest_gene, fitness_history


def run_genetic_algorithm(mapping_desc, metric, number_of_generations, population_size, min_segments=constants.MIN_SEGMENTS,
                          max_segments=constants.MAX_ENGINES_V2):

    board_name = mapping_desc.board_name
    model_name = mapping_desc.model_name
    model_dag = mapping_desc.model_dag
    timing_metric = mapping_desc.timing_metric

    num_conv_layers = utils.get_num_conv_layer_count_in_range(
        model_dag, 0, len(model_dag))
    max_engines = min(constants.MAX_ENGINES_V2, num_conv_layers)

    t0 = time.time()
    population = generate_population(
        population_size, board_name, model_name, model_dag, min_segments, max_engines, timing_metric)
    max_fitness, fittest_gene, fitness_history = optimize_v2(
        population, metric, population_size, number_of_generations)
    duration = time.time() - t0

    timing_file_name = 'ga_population_{}_generations_{}.json'.format(
        population_size, number_of_generations)
    timings_dict = mapping_general_utils.load_json_to_dict(
        os.path.dirname(__file__) + '/timing/' + timing_file_name)

    if timings_dict != None:
        new_timing = duration
        if board_name in timings_dict:
            if model_name in timings_dict[board_name]:
                previous_runs = int(
                    timings_dict[board_name][model_name]['runs'])
                previous_timing = float(
                    timings_dict[board_name][model_name]['duration'])
                previous_timing *= previous_runs
                new_timing = (previous_timing + duration) / (1 + previous_runs)
            else:
                timings_dict[board_name][model_name] = {}
        else:
            timings_dict[board_name] = {model_name: {}}
        if 'runs' not in timings_dict[board_name][model_name]:
            timings_dict[board_name][model_name]['runs'] = 0
        timings_dict[board_name][model_name]['runs'] += 1
        timings_dict[board_name][model_name]['duration'] = new_timing
    else:
        timings_dict = {}
        timings_dict[board_name] = {model_name: {}}
        timings_dict[board_name][model_name]['runs'] = 1
        timings_dict[board_name][model_name]['duration'] = duration

    mapping_general_utils.save_dict_to_json(timings_dict, os.path.dirname(__file__) + '/timing/',
                                            timing_file_name)

    mapping_general_utils.save_dict_to_json(fitness_history, os.path.dirname(__file__) + '/optimizer_progress/{}_{}/'.format(
                                            population_size, number_of_generations),
                                            'ga_progress_{}_{}_{}_population_{}_generations_{}.json'.format(
                                                board_name, model_name,
                                                constants.metric_display_names[metric],
                                                population_size, number_of_generations))

    return max_fitness, fittest_gene
