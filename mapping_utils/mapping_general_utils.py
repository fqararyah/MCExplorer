
import __init__
import copy
import constants
import json
import utils
from preformance_record import *
import math
import os
import csv
from hw_config import *
from collections import defaultdict
from typing import Any, Dict

def swap_levels(d: Dict, level1: int, level2: int) -> Dict:
    def traverse(path, value):
        if not isinstance(value, dict):
            yield path, value
        else:
            for k, v in value.items():
                yield from traverse(path + [k], v)

    # Flatten the dictionary into paths
    flat = list(traverse([], d))

    # Swap specified levels in paths
    swapped_flat = []
    for path, value in flat:
        if level1 < len(path) and level2 < len(path):
            path[level1], path[level2] = path[level2], path[level1]
        swapped_flat.append((tuple(path), value))

    # Reconstruct nested dict
    def insert(tree, path, value):
        for key in path[:-1]:
            tree = tree.setdefault(key, {})
        tree[path[-1]] = value

    result = {}
    for path, value in swapped_flat:
        insert(result, path, value)

    return result

def generate_distributions(n, k):
    def helper(n, k, current, result):
        # If we've placed all items and have no more bags to fill
        if k == 0:
            if n == 0:
                result.append(tuple(current))
            return

        # Try placing at least one item in each bag
        # i must be at least 1 and we ensure there's enough items for the remaining bags
        for i in range(1, n - k + 2):
            helper(n - i, k - 1, current + [i], result)

    result = []
    helper(n, k, [], result)
    return result


def prepare_save_path_from_list(base_dir, dir_list):
    save_path = base_dir + '/'
    for i in range(len(dir_list)):
        save_path += dir_list[i] + '/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    return save_path

def proportional_allocation(num_pes, op_counts_list):

    engines_pes = []
    alpha = 0
    sum_roots = 0
    for i in range(len(op_counts_list)):
        sum_roots += math.sqrt(op_counts_list[i])

    alpha = (sum_roots / num_pes) ** 2

    for i in range(len(op_counts_list)):
        engines_pes.append(max(1, int(math.sqrt(op_counts_list[i] / alpha))))

    return engines_pes

def dec_to_binary_as_list(num, num_digits = -1):
    current_digit_position = 0
    if num_digits == -1:
        num_digits = int(math.log2(num) + 1)
    binary_list = [0] * num_digits
    while num > 0:
        binary_list[current_digit_position] = num % 2
        current_digit_position += 1
        num //= 2

    return binary_list


def prepare_save_path(base_dir, relative_save_path):
    splits = relative_save_path.split('/')

    return prepare_save_path_from_list(base_dir, splits)


def prepare_save_path_abs(path_str):
    tmp_path = '/' if path_str[0] == '/' else './'
    path_str_splits = path_str.split('/')

    for split in path_str_splits:
        tmp_path += split
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)
        tmp_path += '/'


def overleaf_table_format(data):
    out = "\\begin{}{}{} \\hline\n".format(
        '{tabular}', '{|l|' + ('c|' * len(data[0])), '}')
    for i in range(len(data)):
        for j in range(len(data[i])):
            out += str(data[i][j]) + (' & ' if j < len(data[i]) - 1 else '')
        out += '\\\\ \\hline \n'

    out += '\end{tabular}'

    return out


def read_board_names(file_name=constants.HW_CONFIGS_FILE):
    board_names = []
    f = open(file_name)
    content = json.load(f)
    for entry in content:
        board_names.append(entry['board_name'])

    return board_names


def highest_power_of_2_divides_num(num):
    if num == 1:
        return 1

    return (num & (~(num - 1)))


def calc_list_gcd(list):
    list_gcd = math.gcd(list[0], list[1])
    for i in range(2, len(list)):
        list_gcd = math.gcd(list_gcd, list[i])

    return list_gcd


def list_divided_by_its_gcd(list):
    list_gcd = calc_list_gcd(list)
    list_by_its_gcd = [list[k] / list_gcd for k in range(len(list))]

    return list_by_its_gcd

# Maximal Disjoint Intervals


def max_segments_of_same_balance(model_dag, segment_len):
    layers_ops = utils.get_layers_op_counts(model_dag)
    layers_indices = utils.get_conv_layer_indices_in_range(
        model_dag, 0, len(model_dag))
    max_non_overlaping_balanced_segments = []
    for i in range(len(layers_ops) - 2 * segment_len):
        j = i + segment_len
        base_segment_ops = layers_ops[i:i+segment_len]
        base_segment_ops_by_gcd = list_divided_by_its_gcd(base_segment_ops)
        non_overlaping_balanced_segments = [i]
        while j < len(layers_ops) - segment_len:
            current_segment_ops = layers_ops[j:j+segment_len]
            current_segment_ops_by_gcd = list_divided_by_its_gcd(
                current_segment_ops)
            imbalance_list = [max(a, b) / min(a, b)
                              for a, b in zip(base_segment_ops_by_gcd, current_segment_ops_by_gcd)]
            if max(imbalance_list) == 1:
                non_overlaping_balanced_segments.append(j)
                j += segment_len
            else:
                j += 1
        if len(non_overlaping_balanced_segments) > 1:
            max_non_overlaping_balanced_segments.append(
                copy.deepcopy(non_overlaping_balanced_segments))

    for i in range(len(max_non_overlaping_balanced_segments)):
        for j in range(len(max_non_overlaping_balanced_segments[i])):
            max_non_overlaping_balanced_segments[i][j] = layers_indices[max_non_overlaping_balanced_segments[i][j]]

    return max_non_overlaping_balanced_segments


def build_performance_record(mapping, board_name, model_name, num_engines):

    estimated_exec_time = round(1000 * mapping.calc_exec_time(), 2)
    estimated_throughput = mapping.calc_throughput()

    on_chip_fms_buffer_sz = round(
        mapping.calc_fms_buffer_sz() / constants.MiB, 2)
    on_chip_weights_buffer_sz = round(
        mapping.calc_weights_buffer_sz() / constants.MiB, 2)
    on_chip_buffer_sz = round(
        mapping.calc_on_chip_buffer_sz() / constants.MiB, 2)

    off_chip_weight_access = round(
        mapping.calc_off_chip_weights_access() / constants.MiB, 2)
    off_chip_fms_access = round(
        mapping.calc_off_chip_fms_access() / constants.MiB, 2)
    off_chip_access = off_chip_weight_access + off_chip_fms_access

    return PerformanceRecord(board_name, model_name, mapping.get_label(), num_engines,
                             estimated_exec_time, estimated_throughput,
                             on_chip_fms_buffer_sz, on_chip_weights_buffer_sz,
                             on_chip_buffer_sz,
                             off_chip_fms_access, off_chip_weight_access)


def get_bram_block_used_byes(data_bit_width):
    return int(constants.BRAM_BLOCK_KBITS / data_bit_width) * constants.KiB


def read_mappings_json(mappings_file):
    mapping_list = []
    with open(mappings_file) as f:
        content = json.load(f)
    for entry in content:
        mapping_list.append(entry)

    return mapping_list


def clean_engine_range(range_str):
    if '-' in range_str:
        splits = range_str.split('-')
        first_engine_index = int(splits[0])
        last_engine_index = int(splits[1])

        range_list = [*range(first_engine_index, last_engine_index)]
    else:
        range_list = [int(range_str)]

    return range_list


def clean_layer_range(range_str, model_dag):
    if '-' in range_str:
        splits = range_str.split('-')
        first_layer_offset = int(splits[0])
        first_layer_index = utils.get_conv_layer_index_from_offset(
            model_dag, 0, first_layer_offset)
        if splits[1] == 'last':
            last_layer_offset = utils.get_num_conv_layer_count_in_range(
                model_dag, 0, len(model_dag))
        else:
            last_layer_offset = int(splits[1])

        num_layers = last_layer_offset - first_layer_offset

        range_list = utils.get_conv_layer_indices_in_range(
            model_dag, first_layer_index, num_layers)
    else:
        range_list = [utils.get_conv_layer_index_from_offset(
            model_dag, 0, int(range_str))]

    return range_list


def get_model_display_name(model_name):
    return constants.model_display_names[model_name]


def get_model_display_name_list(model_name_list):
    display_model_name_list = []
    for model_name in model_name_list:
        display_model_name_list.append(
            constants.model_display_names[model_name] if model_name in constants.model_display_names else model_name)

    return display_model_name_list


def get_board_display_name_list(board_name_list):
    display_board_name_list = []
    for board_name in board_name_list:
        display_board_name_list.append(board_name.upper())

    return display_board_name_list


def save_dict_to_json(dict_to_save, file_path, file_name: str):
    prepare_save_path_abs(file_path)
    json_obj = json.dumps(dict_to_save)
    if not file_name.endswith('.json'):
        file_name += '.json'
    with open(file_path + file_name, 'w') as f:
        f.write(json_obj)


def load_json_to_dict(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        return json.load(f)


def validate_mapping_dict(mapping_dict, model_dag):

    expected_first_layer = 0
    expected_first_engine = 0
    num_conv_layers = utils.get_num_conv_layer_count_in_range(
        model_dag, 0, len(model_dag))
    model_last_layer = 0
    for layers, engines in mapping_dict.items():
        if '-' in layers:
            splits = layers.split('-')
            first_layer = int(splits[0])
            last_layer = int(splits[1])
            model_last_layer = last_layer
            if first_layer != expected_first_layer:
                print('layers: expected {}, but found {}'.format(
                    expected_first_layer, first_layer))
                return False
            expected_first_layer = last_layer
        else:
            single_layer = int(layers)
            model_last_layer = single_layer
            if single_layer != expected_first_layer:
                print('layer: expected {}, but found {}'.format(
                    expected_first_layer, single_layer))
                return False
            expected_first_layer = single_layer + 1
        if '-' in engines:
            splits = engines.split('-')
            first_engine = int(splits[0])
            last_engine = int(splits[1])
            if first_engine != expected_first_engine:
                print('engines: expected {}, but found {}'.format(
                    expected_first_engine, first_engine))
                return False
            expected_first_engine = last_engine
        else:
            single_engine = int(engines)
            if single_engine != expected_first_engine:
                print('engine: expected {}, but found {}'.format(
                    expected_first_engine, single_engine))
                return False
            expected_first_engine = single_engine + 1

    if model_last_layer < num_conv_layers - 1:
        print('missing layers, expected last layer to be {} but found {}'.format(
            num_conv_layers - 1, model_last_layer))
        return False

    return True


def select_with_decreased_propability_thresholds(num_elements):
    threshold_list = [0] * num_elements
    sum_element_thresholds = num_elements * (num_elements + 1) / 2
    starting_threshold = num_elements
    acc_threshold = 0

    for i in range(num_elements):
        acc_threshold += starting_threshold / sum_element_thresholds
        threshold_list[i] = acc_threshold
        starting_threshold -= 1

    return threshold_list


def select_with_decreased_propability(num_elements, rand_val, thresholds_list=None):
    """
    assume a list of 5 elements, the goal is to select the elements
    towards the beginning with higher propabilities, i.e.:
    the propabilities of selecting the elements are [5x, 4x, 3x, ...]
    5x + 4x + ... = 1, hence, x = 1 / (5 + 4 + ...)
    """
    if thresholds_list is None:
        thresholds_list = select_with_decreased_propability_thresholds(
            num_elements)

    for i in range(num_elements):
        if rand_val <= thresholds_list[i]:
            return i

    return num_elements - 1


def get_latest_file_path(file_path, file_name_template):

    splited_name = file_name_template.split('.')
    file_name_prefix = splited_name[0]
    file_extention = splited_name[1]
    last_file_name = 'DOES_NOT_EXIST'
    print('>', file_name_prefix)
    for file_name in os.listdir(file_path):
        if file_name_prefix in file_name and file_extention in file_name \
                and file_name > last_file_name:
            last_file_name = file_name

    return file_path + last_file_name


def get_metric_values_from_perf_records(perf_record_list, metric):
    metric_list = []
    for perf_record in perf_record_list:
        metric_list.append(perf_record.get_metric_val(metric))

    return metric_list

def adjust_mapping_dict_layers(mapping_dict, new_layer_lists):
    new_mapping_dict = {}
    layers_so_far = 0
    i = 0
    for _ ,engines in mapping_dict.items():
        current_num_layers = len(new_layer_lists[i])
        i += 1
        if current_num_layers == 1:
            new_mapping_dict['{}'.format(layers_so_far)] = engines
        else:
            new_mapping_dict['{}-{}'.format(layers_so_far, layers_so_far + current_num_layers)] = engines
        layers_so_far += current_num_layers
    
    return new_mapping_dict

def segment_layers_indices_from_mapping_dict(mapping_dict, model_dag):
    segment_layers_dict = {}
    segment_index = 0
    for layers, _ in mapping_dict.items():
        layer_lits = clean_layer_range(layers, model_dag)
        segment_layers_dict[segment_index] = layer_lits
        segment_index += 1

    return segment_layers_dict

def segment_layers_indices_ranges_from_mapping_dict(mapping_dict, model_dag):
    segment_layers_dict = {}
    segment_index = 0
    for layers, _ in mapping_dict.items():
        layer_lits = clean_layer_range(layers, model_dag)
        segment_layers_dict[segment_index] = (layer_lits[0], layer_lits[-1])
        segment_index += 1

    return segment_layers_dict


def segment_layers_op_counts_from_mapping_list(mapping_dict, model_dag):
    segment_layers_ops_list = []
    segment_index = 0
    for layers, _ in mapping_dict.items():
        layer_lits = clean_layer_range(layers, model_dag)
        segment_layers_ops_list.append(utils.get_layers_op_counts_by_indices(
            model_dag, layer_lits))
        segment_index += 1

    return segment_layers_ops_list


def get_metric_values_from_perf_records_dicts(perf_record_dicts_list, metric):
    perf_record_list = []
    for perf_dict in perf_record_dicts_list:
        perf_record = PerformanceRecord()
        perf_record.init_from_dict(perf_dict)
        perf_record_list.append(perf_record)

    return get_metric_values_from_perf_records(perf_record_list, metric)


def copy_2d_list(inp_list):
    a_copy = []
    for sub_list in inp_list:
        a_copy.append([])
        for item in sub_list:
            a_copy[-1].append(item)

    return a_copy


def csv_to_dict(file_path):
    ret_dict = {}
    with open(file_path, mode='r')as file:
        csvFile = csv.reader(file)
        column_keys = []
        line_num = 0
        for lines in csvFile:
            for iteme_index, item in enumerate(lines):
                if line_num == 0:
                    ret_dict[item] = []
                    column_keys.append(item)
                else:
                    if '.' in item:
                        item = float(item)
                    else:
                        item = int(item)

                    ret_dict[column_keys[iteme_index]].append(item)

            line_num += 1

    return ret_dict


def approximate_power(power_dict, num_hw_units):

    x_list = power_dict['units']
    y_list = power_dict['power']

    if num_hw_units in x_list:
        index = x_list.index(num_hw_units)
        return y_list[index]

    x1 = 0
    y1 = 0
    for index in range(len(x_list)):
        x2 = x_list[index]
        y2 = y_list[index]
        if x2 > num_hw_units:
            # print(y1, (y2 - y1), (num_hw_units - x1),  (x2 - x1))
            return y1 + (y2 - y1) * (num_hw_units - x1) / (x2 - x1)

        x1 = x2
        y1 = y2

    return y1

def get_ideal_throughput(hw_config_obj, model_dag):

    op_count = sum(utils.get_layers_op_counts(model_dag, 0))
    pes = hw_config_obj.num_pes
    ideal_runtime = op_count / (pes * hw_config_obj.frequency) 

    return 1 / ideal_runtime

def get_ideal_latency(hw_config_obj, model_dag):
    op_counts_list = utils.get_layers_op_counts(model_dag, 0)
    op_count = sum(op_counts_list)

    ideal_runtime = op_count
    pes = hw_config_obj.num_pes
    best_split = 1
    best_pes = []
    for split in range(1, len(op_counts_list) - 1):
        op_counts_1 = sum(op_counts_list[0: split])
        op_counts_2 = sum(op_counts_list[split:])
        engines_pes = proportional_allocation(pes, [op_counts_1, op_counts_2])
        current_runtime = (op_counts_1 / engines_pes[0]) + (op_counts_2 / engines_pes[1]) 
        # print(split, (op_counts_1 / engines_pes[0]) / hw_config_obj.frequency , 
        #       (op_counts_2 / engines_pes[1]) / hw_config_obj.frequency  , engines_pes[0], engines_pes[1])
        if current_runtime < ideal_runtime:
            ideal_runtime = current_runtime
            best_split = split
            best_pes = engines_pes

    ideal_runtime = 1000 * ideal_runtime / hw_config_obj.frequency 

    return ideal_runtime, best_split, best_pes

def get_ideal(hw_config_obj, model_dag, metric):
    if metric == Metrics.LATENCY:
        ideal_latency, _, _ = get_ideal_latency(hw_config_obj, model_dag)
        return ideal_latency
    elif metric == Metrics.THROUGHPUT:
        return get_ideal_throughput(hw_config_obj, model_dag)