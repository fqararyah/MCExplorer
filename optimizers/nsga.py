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
mutation_probability = 0.4
elite_percentage = 0.0
mutable_elite_percentage = 0.0
tournament_size = 2
min_population_size = 40
max_occurrences_percentage = 0.04

class Individual:
    def __init__(self, objective_list, metric_list, mapping_desc):
        self.objective_list = objective_list  # List of objectives
        self.metric_list = metric_list
        self.mapping_desc = mapping_desc
        self.rank = None  # Pareto front rank
        self.dominated_count = 0  # How many individuals dominate this one
        self.dominated_set = []  # List of individuals that are dominated by this one
        self.crowding_distance = 0

    def __hash__(self):
        return hash(self.mapping_desc.__str__())
    
    def __eq__(self, other):
        if isinstance(other, Individual):
            return self.mapping_desc.__str__() == other.mapping_desc.__str__()
        return False
    
    def copy_individual(self):
        a_copy = Individual(copy.copy(self.objective_list), 
                            copy.copy(self.metric_list), self.mapping_desc.copy_mapping_desc())
        a_copy.rank = self.rank
        a_copy.dominated_count = self.dominated_count
        new_dominated_set = []
        for ind in self.dominated_set:
            new_dominated_set.append(copy.copy(ind))
        a_copy.dominated_set = copy.copy(new_dominated_set)
        a_copy.crowding_distance = self.crowding_distance
        
        return a_copy


# the objective function of six hump camelback
def member_objectives(mapping, metric_list):
    objectives = []
    for metric in metric_list:
        if metric == Metrics.LATENCY:
            objectives.append(mapping.calc_exec_time())
        elif metric == Metrics.THROUGHPUT:
            objectives.append(mapping.calc_throughput())
        elif metric == Metrics.BUFFER:
            objectives.append(mapping.calc_on_chip_buffer_sz())
        elif metric == Metrics.ACCESS:
            objectives.append(mapping.calc_off_chip_weights_access(
            ) + mapping.calc_off_chip_fms_access())
        elif metric == Metrics.ENERGY:
            objectives.append(mapping.calc_energy())
        elif metric == Metrics.REQUIED_BW:
            objectives.append( (mapping.calc_off_chip_weights_access(
            ) + mapping.calc_off_chip_fms_access() ) /  mapping.calc_exec_time())

    return objectives


def generate_population(population_size, board_name, model_name, model_dag, min_segments, max_engines,
                        metric_list, timing_metric):
    population = []
    ce_possibilities = max_engines - min_segments + 1

    population_per_ce_count = (
        population_size + ce_possibilities - 1) // ce_possibilities

    for num_segments in range(min_segments, max_engines + 1):
        for j in range(population_per_ce_count):
            adjust_pes = random.random() <= 0.5
            if len(population) >= population_size:
                break
            if j == 0:
                mapping_desc_dict = {'{}-{}'.format(0, num_segments): '{}-{}'.format(0, num_segments),
                                     '{}-last'.format(num_segments): '{}'.format(num_segments)}
                mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
                        board_name, model_dag, mapping_desc_dict, timing_metric, adjust_pes = adjust_pes)
                mapping_desc = MappingDescription(board_name, model_name, model_dag, timing_metric=timing_metric)
                mapping_desc.mapping_desc_from_dict(mapping_desc_dict)
            else:
                mapping, mapping_desc =\
                    custom_mapping_utils.generate_random_mapping(
                        board_name, model_name, model_dag, num_segments, timing_metric)

            objective_list = member_objectives(mapping, metric_list)
            population.append(Individual(
                objective_list, metric_list, mapping_desc))

    return population


def dominates(ind1, ind2):
    """
    Checks if individual ind1 dominates individual ind2.
    A dominates B if A is better than B in at least one objective and no worse in others.
    """
    dominates = False
    for i in range(len(ind1.objective_list)):
        if first_better_than_second(ind1.metric_list[i], ind2.objective_list[i], ind1.objective_list[i]):
            return False
        if first_better_than_second(ind1.metric_list[i], ind1.objective_list[i], ind2.objective_list[i]):
            dominates = True
    return dominates


def non_dominated_sort(population):
    """
    Perform non-dominated sorting on a population of individuals.
    Returns a list of Pareto fronts.
    """
    fronts = [[]]  # List of Pareto fronts
    for p in population:
        p.dominated_count = 0
        p.dominated_set = []
        for q in population:
            if dominates(p, q):
                p.dominated_set.append(q)
            elif dominates(q, p):
                p.dominated_count += 1

        # If the individual is not dominated by anyone, it belongs to the first front
        if p.dominated_count == 0:
            p.rank = 0
            fronts[0].append(p)

    current_front = 0
    while fronts[current_front]:
        next_front = []
        for p in fronts[current_front]:
            for q in p.dominated_set:
                q.dominated_count -= 1
                if q.dominated_count == 0:
                    q.rank = current_front + 1
                    next_front.append(q)

        current_front += 1
        fronts.append(next_front)

    return fronts


def crowding_distance(front):
    if len(front) == 0:
        return

    num_objectives = len(front[0].objective_list)
    for i in range(num_objectives):
        front.sort(key=lambda x: x.objective_list[i])

        # Set the crowding distance for boundary individuals to infinity
        front[0].crowding_distance = float('inf')
        front[-1].crowding_distance = float('inf')

        # Calculate crowding distance for each individual
        for j in range(1, len(front) - 1):
            front[j].crowding_distance += (front[j + 1].objective_list[i] - front[j - 1].objective_list[i]) / \
                (front[-1].objective_list[i] - front[0].objective_list[i] + 1)


def select_next_generation(population, offspring, population_size):
    # Combine parent and offspring populations
    combined_population = population + offspring

    # Perform non-dominated sorting
    fronts = non_dominated_sort(combined_population)

    next_generation = []
    current_front = 0
    occurrences = {}
    max_occurrences = get_count_from_percentage(max_occurrences_percentage, population_size)
    to_add_if_not_full_list = []
    # Select individuals for the next generation
    while len(next_generation) + len(fronts[current_front]) <= population_size:
        # Add all individuals from the current front
        for individual in fronts[current_front]:
            # Check if the individual has already been selected max_occurrences times
            if individual not in occurrences:
                occurrences[individual] = 0
            if occurrences[individual] < max_occurrences:
                next_generation.append(individual)
                occurrences[individual] += 1
            else:
                to_add_if_not_full_list.append(individual)

        current_front += 1
        if current_front >= len(fronts): 
            break

    # If the next front exceeds the population size, we need to use crowding distance
    if current_front >= len(fronts):
        current_front = 0
    
    while current_front < len(fronts):    
        if len(next_generation) < population_size:
            # Calculate crowding distance for individuals in the current front
            crowding_distance(fronts[current_front])

            # Sort individuals in the current front by crowding distance (larger distance = better)
            fronts[current_front].sort(
                key=lambda x: x.crowding_distance, reverse=True)

            # Add individuals from the current front, based on crowding distance, until population size is met
            next_generation.extend(
                fronts[current_front][:population_size - len(next_generation)])

        current_front += 1
            
    return next_generation


def get_count_from_percentage(percentage, population_size):
    count = int(percentage * population_size)
    if percentage > 0 and count == 0:
        count = 1

    return count

def get_elite_count_from_percentage(percentage, population_size):
    elite_count = int(percentage * population_size)
    if percentage > 0 and elite_count % 2 != 0:
        elite_count += 1
    if percentage != 0 and elite_count == 0:
        elite_count = 2

    return elite_count


def update_population_objectives(population, offsprings, population_size, metric_list):
    num_objectives = len(population[0].objective_list)
    max_objectives = [0] * num_objectives

    for i in range(population_size):
        mapping_desc = population[i].mapping_desc
        mapping = custom_mapping_utils.custom_mapping_from_desc(mapping_desc)
        objective_list = member_objectives(mapping, metric_list)
        for j in range(num_objectives):
            max_objectives[j] = max(max_objectives[j], objective_list[j])
        population[i].objective_list = objective_list

    for i in range(population_size):
        mapping_desc = offsprings[i].mapping_desc
        mapping = custom_mapping_utils.custom_mapping_from_desc(mapping_desc)
        objective_list = member_objectives(mapping, metric_list)
        for j in range(num_objectives):
            max_objectives[j] = max(max_objectives[j], objective_list[j])
        offsprings[i].objective_list = objective_list

    for i in range(population_size):
        for j in range(num_objectives):
            population[i].objective_list[j] /= max_objectives[j]
            offsprings[i].objective_list[j] /= max_objectives[j]


def sort_population_by_segment_count(population):
    population_engine_map = {}
    for member in population:
        if member.mapping_desc.num_segments not in population_engine_map:
            population_engine_map[member.mapping_desc.num_segments] = []
        population_engine_map[member.mapping_desc.num_segments].append(member)

    i = 0
    for num_engines, members in population_engine_map.items():
        for member in members:
            population[i] = member
            i += 1

    return population


def do_crossover(parent_list, population_size):
    offspring_list = []

    elite_count = get_elite_count_from_percentage(
        elite_percentage, population_size)
    for i in range(elite_count):
        offspring_list.append(parent_list[i].copy_individual())

    itr = elite_count
    while (itr < population_size - 1):
        parent1_index = itr
        parent2_index = (itr+1)
        parent_1 = parent_list[parent1_index].copy_individual()
        parent_2 = parent_list[parent2_index].copy_individual()

        x = random.random()
        if x > crossover_probability or parent_1.mapping_desc.num_segments != parent_2.mapping_desc.num_segments \
                or parent_1.mapping_desc.num_segments == 1:
            itr += 2
            offspring_list.append(parent_1)
            offspring_list.append(parent_2)
            continue

        parent_1_desc = parent_1.mapping_desc
        parent_2_desc = parent_2.mapping_desc

        num_segments = parent_1_desc.num_segments
        if num_segments == 2:
            crossover_point = 1
        else:
            crossover_point = random.randint(1, num_segments - 2)
        _, offspring1_mapping_desc = custom_mapping_utils.mapping_from_crossover_of_two_mappings(
            parent_1_desc, parent_2_desc, crossover_point)
        offspring_list.append(Individual(
            parent_1.objective_list, parent_1.metric_list, offspring1_mapping_desc))
        _, offspring2_mapping_desc = custom_mapping_utils.mapping_from_crossover_of_two_mappings(
            parent_2_desc, parent_1_desc, crossover_point)
        offspring_list.append(Individual(
            parent_2.objective_list, parent_2.metric_list, offspring2_mapping_desc))

        itr += 2

    return offspring_list


def do_mutation(offspring_list, population_size):
    elite_count = get_elite_count_from_percentage(elite_percentage, population_size)
    for i in range(elite_count, population_size):
        random_value = random.random()
        if random_value > mutation_probability:
            continue
        ind = offspring_list[i].copy_individual()
        mapping_desc = ind.mapping_desc
        _, mutated_desc = custom_mapping_utils.modeify_mapping_random(
            mapping_desc)
        offspring_list[i] = Individual(
            ind.objective_list, ind.metric_list, mutated_desc)

    return offspring_list


def print_population(population):
    for ind in population:
        mapping_desc = ind.mapping_desc
        print(custom_mapping_utils.prepare_custom_mapping_desc(
            mapping_desc.segment_layers_list,
            mapping_desc.segment_block_list,
            mapping_desc.block_engines_list), ind.objective_list)
    print('*******************************')


def optimize(population, metric_list, population_size, number_of_generations, print_timing = False):
    #print_population(population)
    for i in tqdm(range(number_of_generations)):
        # for i in range(number_of_generations):
        # population = get_fittest_parents_roulette_wheel(fitness, population, population_size, metric)
        if print_timing:
            t0 = time.time()
        offsprings = do_crossover(population, population_size)
        if print_timing:
            print('do_crossover', time.time() - t0)

        if print_timing:
            t0 = time.time()
        offsprings = do_mutation(offsprings, population_size)
        if print_timing:
            print('do_mutation', time.time() - t0)

        if print_timing:
            t0 = time.time()
        update_population_objectives(
            population, offsprings, population_size, metric_list)
        if print_timing:
            print('update_population_objectives', time.time() - t0)

        if print_timing:
            t0 = time.time()
        population = select_next_generation(
            population, offsprings, population_size)
        if print_timing:
            print('select_next_generation', time.time() - t0)
        
        if print_timing:
            t0 = time.time()
        sort_population_by_segment_count(population)
        if print_timing:
            print('sort_population_by_segment_count', time.time() - t0)
        #print_population(population)

        # population, population_size = adaptive_shrinking(
        #    population, fitness, max_fitness, metric, population_size)

    return population


def run_nsga2(mapping_desc, metric_list, number_of_generations, population_size, min_segments=constants.MIN_SEGMENTS,
              max_segments=constants.MAX_ENGINES_V2, print_timing = False):

    board_name = mapping_desc.board_name
    model_name = mapping_desc.model_name
    model_dag = mapping_desc.model_dag
    timing_metric = mapping_desc.timing_metric

    num_conv_layers = utils.get_num_conv_layer_count_in_range(
        model_dag, 0, len(model_dag))
    max_engines = min(constants.MAX_ENGINES_V2, num_conv_layers)

    t0 = time.time()
    population = generate_population(
        population_size, board_name, model_name, model_dag, min_segments, max_engines, metric_list, timing_metric)

    population = optimize(
        population, metric_list, population_size, number_of_generations, print_timing=print_timing)
    
    duration = time.time() - t0

    timing_file_name = 'nsga_population_{}_generations_{}.json'.format(
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

    to_retern_population = []
    for ind in population:
        to_retern_population.append(ind.mapping_desc)

    return to_retern_population
