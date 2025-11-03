import __init__
import constants as consts
from preformance_record import *
from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping import *
from mapping_types.segment_grained_mapping_rr import *
import mapping_utils.mapping_general_utils as mapping_general_utils
import mapping_utils.mapping_exec_utils as mapping_exec_utils
import os
from datetime import datetime


def get_bests_in_all_base_mappings_and_metrics(board_name_list, model_name_list, run_anyway=False, 
                                               custom_dir_postfix = None):

    mapping_label_list = constants.mappings_ordered
    metric_list = [Metrics.LATENCY, Metrics.THROUGHPUT, Metrics.ENERGY]

    metrics_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        metric_list)
    boards_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        board_name_list)
    models_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        model_name_list)

    if custom_dir_postfix is None or custom_dir_postfix == '':
        json_file_name = '/baselines/bests_in_base_mappings_metrics_{}_{}_{}.json'.format(
            metrics_str, boards_str, models_str)
    else:
      json_file_name = '/baselines/{}/bests_in_base_mappings_metrics_{}_{}_{}.json'.format(
            custom_dir_postfix, metrics_str, boards_str, models_str)  
    json_file_path = constants.FIGURES_DATA_DIR_P2 + json_file_name
    if not os.path.exists(json_file_path) or run_anyway:
        best_mappings_dict = mapping_exec_utils.get_bests_of_mappings(board_name_list, model_name_list, mapping_label_list,
                                                                      constants.MIN_ENGINES, constants.MAX_ENGINES_V2, metric_list)

        mapping_general_utils.save_dict_to_json(
            best_mappings_dict, constants.FIGURES_DATA_DIR_P2, json_file_name)

    tmp_dict = mapping_general_utils.load_json_to_dict(json_file_path)
    ret_dict = {}
    for metric in metric_list:
        metric_label = consts.metric_display_names[metric]
        ret_dict[metric_label] = {}
        for board_name in board_name_list:
            ret_dict[metric_label][board_name] = {}
            for model_name in model_name_list:
                if model_name not in tmp_dict[metric_label][board_name]:
                    continue
                ret_dict[metric_label][board_name][model_name] = {}
                for mapping_label in mapping_label_list:
                    ret_dict[metric_label][board_name][model_name][mapping_label] = tmp_dict[metric_label][board_name][model_name][mapping_label]
                    
    return ret_dict
    


def get_base_mappings_performance_all_boards_models(run_anyway=False, pes=False,
                                                    large_mem=False, max_engines_bounded_by_layers=False,
                                                    metric_list=None):

    if metric_list is None:
        metric_list = [Metrics.LATENCY, Metrics.THROUGHPUT, Metrics.ENERGY]

    metrics_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        metric_list)

    model_name_list = consts.model_names
    if pes:
        if large_mem:
            board_name_list = mapping_general_utils.read_board_names(
                constants.HW_CONFIGS_FILE_PEs_LARGE_MEM)
        else:
            board_name_list = mapping_general_utils.read_board_names(
                constants.HW_CONFIGS_FILE_PEs_LIMITED_MEM)
    else:
        board_name_list = mapping_general_utils.read_board_names(
            constants.HW_CONFIGS_FILE)
    model_name_list = constants.model_names

    boards_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        board_name_list)
    models_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        model_name_list)

    if max_engines_bounded_by_layers:
        max_ces_str = 'un'
    else:
        max_ces_str = str(constants.MAX_ENGINES)

    json_file_name = 'base_mappings_{}_{}_{}_max_ces_{}.json'.format(metrics_str,
                                                                     boards_str, models_str, max_ces_str)

    json_file_path = constants.FIGURES_DATA_DIR_P2 + '/baselines/' + json_file_name

    print(json_file_path)
    perf_dict = {}
    if not os.path.exists(json_file_path) or run_anyway:
        for board_name in board_name_list:
            perf_dict[board_name] = {}
            for model_name in model_name_list:
                print(board_name, model_name)
                perf_dict[board_name][model_name] = mapping_exec_utils.run_base_mappings_engine_range(
                    board_name, model_name,
                    max_engines_bounded_by_layers=max_engines_bounded_by_layers,
                    min_engines=constants.MIN_ENGINES, max_engines=constants.MAX_ENGINES,
                    serialize_records=True)

        mapping_general_utils.save_dict_to_json(
            perf_dict, constants.FIGURES_DATA_DIR_P2 + '/baselines/', json_file_name)

    return mapping_general_utils.load_json_to_dict(json_file_path)


def get_bests_in_all_base_mappings_boards_models_and_metrics(run_anyway=False, pes=False, custom_dir_postfix = ''):
    if pes:
        board_name_list = mapping_general_utils.read_board_names(
            constants.HW_CONFIGS_FILE_PEs)
    else:
        board_name_list = mapping_general_utils.read_board_names(
            constants.HW_CONFIGS_FILE)
        
    model_name_list = constants.model_names
    #model_name_list = ['cmt', 'unet']
    #print(model_name_list)

    return get_bests_in_all_base_mappings_and_metrics(board_name_list, model_name_list, run_anyway, custom_dir_postfix = custom_dir_postfix)


def get_simulated_annealing_experiments_all_metrics_boards_models(num_iterations,
                                                                  metric_list=None,
                                                                  run_anyway=False,
                                                                  board_name_list = None,
                                                                  model_name_list = None,
                                                                  custom_dir_postfix = '',
                                                                  file_name = None):
    if board_name_list is None:
        board_name_list = mapping_general_utils.read_board_names(
            constants.HW_CONFIGS_FILE)
    if model_name_list is None:
        model_name_list = consts.model_names
    if metric_list is None:
        metric_list = [Metrics.LATENCY, Metrics.THROUGHPUT, Metrics.ENERGY]

    metrics_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        metric_list)
    boards_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        board_name_list)
    models_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        model_name_list)

    json_file_name = file_name
    if file_name is None:
        json_file_name = 'simulated_annealing_bests_in_metrics_{}_boards_{}_models_{}_{}.json'.format(
            metrics_str, boards_str, models_str,
            num_iterations)

    sub_dir = '/simulated/'
    if custom_dir_postfix != '':
        sub_dir += custom_dir_postfix + '/'
    
    print(sub_dir, json_file_name)
    os.makedirs(constants.FIGURES_DATA_DIR_P2 + sub_dir, exist_ok=True)
    json_file_path = mapping_general_utils.get_latest_file_path(
        constants.FIGURES_DATA_DIR_P2 + sub_dir, json_file_name)
    print(json_file_path)
    if run_anyway or not os.path.exists(json_file_path):
        best_mappings_dict = mapping_exec_utils.run_simulated_annealing_experiments(board_name_list, model_name_list,
                                                                                    metric_list, num_iterations)
        mapping_general_utils.save_dict_to_json(best_mappings_dict, constants.FIGURES_DATA_DIR_P2 + sub_dir,
                                                json_file_name)

    return mapping_general_utils.load_json_to_dict(json_file_path)


def get_genetic_algorithm_experiments_all_metrics_boards_models(number_of_generations, population_size,
                                                                metric_list=None,
                                                                run_anyway=False, 
                                                                board_name_list = None,
                                                                model_name_list = None,
                                                                custom_dir_postfix = '', 
                                                                file_name = None):
    if board_name_list is None:
        board_name_list = mapping_general_utils.read_board_names(
            constants.HW_CONFIGS_FILE)
    if model_name_list is None:
        model_name_list = consts.model_names

    if metric_list is None:
        metric_list = [Metrics.LATENCY, Metrics.THROUGHPUT, Metrics.ENERGY]

    metrics_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        metric_list)
    boards_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        board_name_list)
    models_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        model_name_list)

    json_file_name = file_name
    if file_name is None:
        json_file_name = ('genetic_algorithm_metrics_{}_boards_{}_models_{}_' +
                        'population_{}_generations_{}.json').format(
            metrics_str, boards_str, models_str,
            population_size,
            number_of_generations)

    sub_dir = '/genetic/'
    if custom_dir_postfix != '':
        sub_dir += custom_dir_postfix + '/'
    os.makedirs(constants.FIGURES_DATA_DIR_P2 + sub_dir, exist_ok=True)
    json_file_path = mapping_general_utils.get_latest_file_path(
        constants.FIGURES_DATA_DIR_P2 + sub_dir, json_file_name)

    if run_anyway or not os.path.exists(json_file_path):
        best_mappings_dict = mapping_exec_utils.run_genetic_algorithm_experiments(board_name_list,
                                                                                  model_name_list, metric_list,
                                                                                  number_of_generations, population_size)

        mapping_general_utils.save_dict_to_json(best_mappings_dict, constants.FIGURES_DATA_DIR_P2 + sub_dir,
                                                json_file_name)

    return mapping_general_utils.load_json_to_dict(json_file_path)


def get_ideal_values_all_metrics_boards_models(metric_list=None, ref_dict = None):
    board_name_list = mapping_general_utils.read_board_names(
        constants.HW_CONFIGS_FILE)
    model_name_list = consts.model_names

    if metric_list is None:
        metric_list = [Metrics.LATENCY, Metrics.THROUGHPUT, Metrics.ENERGY]

    ideal_dict = {}
    if ref_dict is not None:
        ref_dict_vals = ref_dict[list(ref_dict.keys())[0]]
    for metric in metric_list:
        metric_label = consts.metric_display_names[metric]
        if ref_dict is not None and metric_label not in ref_dict_vals:
            continue
        ideal_dict[metric_label] = {}
        for board_name in board_name_list:
            if ref_dict is not None and board_name not in ref_dict_vals[metric_label]:
                continue
            ideal_dict[metric_label][board_name] = {}
            hw_config = HWConfig(board_name)
            for model_name in model_name_list:
                if ref_dict is not None and model_name not in ref_dict_vals[metric_label][board_name]:
                    continue
                model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                ideal_val = mapping_general_utils.get_ideal(
                    hw_config, model_dag, metric)
                ideal_dict[metric_label][board_name][model_name] = [
                    ideal_val, ideal_val]

    return ideal_dict


def get_hiera_map_algorithm_experiments_all_metrics_boards_models(max_clusters=constants.MAX_CLUSTERS,
                                                                  metric_list=None,
                                                                  run_anyway=False,
                                                                  model_name_list = None,
                                                                  custom_dir_postfix= '',
                                                                  file_name=None):
    board_name_list = mapping_general_utils.read_board_names(
        constants.HW_CONFIGS_FILE)

    if model_name_list is None:
        model_name_list = consts.model_names
    
    if metric_list is None:
        metric_list = [Metrics.LATENCY, Metrics.THROUGHPUT, Metrics.ENERGY]

    metrics_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        metric_list)
    boards_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        board_name_list)
    models_str = mapping_exec_utils.get_file_prefix_from_metric_board_or_model_list(
        model_name_list)

    json_file_name = file_name
    if file_name is None:
        json_file_name = ('hiera_map_metrics_{}_boards_{}_models_{}_' +
                        'max_clusters_{}.json').format(
            metrics_str, boards_str, models_str,
            max_clusters)

    sub_dir = '/hiera_map/'
    if custom_dir_postfix != '':
        sub_dir += str(custom_dir_postfix) + '/'
    hiera_map_dir = constants.FIGURES_DATA_DIR_P2 + sub_dir

    if not os.path.exists(hiera_map_dir):
        os.mkdir(hiera_map_dir)

    json_file_path = mapping_general_utils.get_latest_file_path(
        hiera_map_dir, json_file_name)

    if run_anyway or not os.path.exists(json_file_path):
        print(json_file_path, json_file_name)
        best_mappings_dict = mapping_exec_utils.run_hiera_map_experiments(board_name_list,
                                                                          model_name_list, metric_list, max_clusters)

        mapping_general_utils.save_dict_to_json(best_mappings_dict, hiera_map_dir,
                                                json_file_name)

    return mapping_general_utils.load_json_to_dict(json_file_path)


def validate_base_mapping_dicts(performance_dict):

    norm_dict = {}
    for metric_label, metric_dict in performance_dict.items():
        metric = consts.display_names_metrics_dict[metric_label]
        norm_dict[metric_label] = {}
        for board_name, board_name_dict in metric_dict.items():
            norm_dict[metric_label][board_name] = {}
            for model_name, model_name_dict in board_name_dict.items():
                model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                for _, values in model_name_dict.items():
                    if not mapping_general_utils.validate_mapping_dict(values[1], model_dag):
                        print('INVALID MAPPING:', values[1])

def add_board_averages(perf_dict):
    
     for metric_label, metric_dict in perf_dict.items():
        for board_name, board_name_dict in metric_dict.items():
            board_sums = {}
            num_models = len(board_name_dict.values())
            for _, model_name_dict in board_name_dict.items():
                for mapping_label, val in model_name_dict.items():
                    if mapping_label not in board_sums:
                        board_sums[mapping_label] = 0
                    board_sums[mapping_label] += val
            perf_dict[metric_label][board_name]['Avg'] = {}
            for mapping_label, val in board_sums.items():
                if 'ideal' in mapping_label.lower():
                    val = 0
                perf_dict[metric_label][board_name]['Avg'][mapping_label] = val / num_models 
    
def normalize_performance_dicts(optimized_dicts, base_dict, include_base=False, saving=False, y_threshold = 0):

    norm_dict = {}
    for metric_label, metric_dict in base_dict.items():
        metric = consts.display_names_metrics_dict[metric_label]
        norm_dict[metric_label] = {}
        for board_name, board_name_dict in metric_dict.items():
            norm_dict[metric_label][board_name] = {}
            for model_name, model_name_dict in board_name_dict.items():
                norm_dict[metric_label][board_name][model_name] = {}
                i = 0
                for mapping_label, values in model_name_dict.items():
                    if i == 0:
                        base_val = values[0]
                        if include_base:
                            norm_dict[metric_label][board_name][model_name][mapping_label] = 1 - y_threshold
                        i = 1
                    else:
                        norm_dict[metric_label][board_name][model_name][mapping_label] = \
                                normalize_metrics(metric, base_val, values[0]) - y_threshold
                for optimization_label, optimized_dict in optimized_dicts.items():
                    if metric_label not in optimized_dict:
                        continue
                    opt_val = optimized_dict[metric_label][board_name][model_name]
                    i = 0
                    for mapping_label, values in model_name_dict.items():
                        if i == 0:
                            base_val = values[0]
                            i = 1
                        norm_dict[metric_label][board_name][model_name][optimization_label] = \
                            (1 - normalize_metrics(metric, base_val, opt_val[0])) if saving else \
                            (normalize_metrics(metric, base_val, opt_val[0]) -  y_threshold)

    return norm_dict


def get_black_box_dicts(optimized_dicts):
    bb_dict = {'MCExplorer': {}}

    for optimization_label, optimized_dict in optimized_dicts.items():
        for metric_label, metric_dict in optimized_dict.items():
            bb_dict['MCExplorer'][metric_label] = {}
            for board_name, board_name_dict in metric_dict.items():
                bb_dict['MCExplorer'][metric_label][board_name] = {}

    for optimization_label, optimized_dict in optimized_dicts.items():
        for metric_label, metric_dict in optimized_dict.items():
            metric = consts.display_names_metrics_dict[metric_label]
            for board_name, board_name_dict in metric_dict.items():
                for model_name, values in board_name_dict.items():
                    if values is None or len(values) == 0:
                        opt_val = -1 if 'thr' in metric_label.lower() else 10000000000
                        values = [opt_val]
                    else:
                        opt_val = values[0]
                    if model_name not in bb_dict['MCExplorer'][metric_label][board_name]:
                        bb_dict['MCExplorer'][metric_label][board_name][model_name] = values
                    elif first_better_than_or_equal_second(metric, opt_val,
                                                           bb_dict['MCExplorer'][metric_label][board_name][model_name][0]):
                        bb_dict['MCExplorer'][metric_label][board_name][model_name] = values

    return bb_dict


def get_best_base_mappings_in_performance_dicts(base_dict):

    bests_dict = {}
    for metric_label, metric_dict in base_dict.items():
        metric = consts.display_names_metrics_dict[metric_label]
        bests_dict[metric_label] = {}
        for board_name, board_name_dict in metric_dict.items():
            bests_dict[metric_label][board_name] = {}
            for model_name, model_name_dict in board_name_dict.items():
                bests_dict[metric_label][board_name][model_name] = {}
                best_val = -1
                best_mapping = None
                best_label = 'bestBase'
                for mapping_label, values in model_name_dict.items():
                    perf_val = values[0]
                    if best_val == -1 or \
                            first_better_than_or_equal_second(metric, perf_val, best_val):
                        best_val = perf_val
                        best_mapping = values[1]
                        # best_label = mapping_label
                bests_dict[metric_label][board_name][model_name][best_label] = [
                    best_val, best_mapping]

    return bests_dict


def get_best_mappings_black_box(metric, population_size, num_generations, num_sa_main_iterations,
                                hm_max_clusters,
                                run_anyway=False):

    ga_dict = get_genetic_algorithm_experiments_all_metrics_boards_models(num_generations, population_size,
                                                                          metric_list=[metric], run_anyway=run_anyway)

    sa_dict = get_simulated_annealing_experiments_all_metrics_boards_models(num_sa_main_iterations,
                                                                            metric_list=[metric], run_anyway=run_anyway)

    hm_dict = get_hiera_map_algorithm_experiments_all_metrics_boards_models(max_clusters=hm_max_clusters,
                                                                            metric_list=[metric], run_anyway=run_anyway)

    optimized_dicts = {'GA': ga_dict, 'SA': sa_dict, 'HM': hm_dict}

    optimized_dicts = get_black_box_dicts(optimized_dicts)

    return optimized_dicts
