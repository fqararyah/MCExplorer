import mapping_utils.mapping_exec_utils as mapping_exec_utils
from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
import mapping_utils.custom_mapping_utils as custom_mapping_utils
import optimizers.simulated_annealing as sa
from preformance_record import *
import experiments_p2.experiments as experiments_p2
import experiments_p2.experiments_utils as experiment_utils
import experiments_p2.ploting_experiments as plot_exps
from plot import plot_results
from plot import plot_utils
from mapping_strategies.mapping_types.mapping_description import *
import optimizers.nsga as nsga2
import multiprocessing
from optimizers import hiera_map
import matplotlib.pyplot as plt
import statistics

population = 400
generations = 50
sa_tenthousands = 20000
hm_max_clusters = constants.MAX_CLUSTERS
run_base_anyway = False
custom_dir_postfix_dirs_list = ['10']


def add_dict_values(src_dict, dst_dict, metric_key):
    if metric_key not in summary_dict:
        dst_dict[metric_key] = {}
    for board_name, board_name_dict in src_dict[metric_key].items():
        if board_name not in dst_dict[metric_key]:
            dst_dict[metric_key][board_name] = {}
        for model_name, values in board_name_dict.items():
            if model_name not in dst_dict[metric_key][board_name]:
                dst_dict[metric_key][board_name][model_name] = []
            val = src_dict[metric_key][board_name][model_name][0] if \
                len(src_dict[metric_key][board_name][model_name]) > 0 else 0
            dst_dict[metric_key][board_name][model_name].append(val)

def mean_stdev(src_dict, mean_dict, std_dict, metric_key):
    mean_dict[metric_key] = {}
    std_dict[metric_key] = {}
    max_cv = 0
    min_cv = 1
    avg_cv = 0
    valid_exps = 0
    for board_name, board_name_dict in src_dict[metric_key].items():
        mean_dict[metric_key][board_name] = {}
        std_dict[metric_key][board_name] = {}
        for model_name, values in board_name_dict.items():
            if len(src_dict[metric_key][board_name][model_name]) < 2:
                continue
            mean = statistics.mean(src_dict[metric_key][board_name][model_name])
            mean_dict[metric_key][board_name][model_name] = [mean]
            std = statistics.stdev(src_dict[metric_key][board_name][model_name])
            cv = (std / mean) if mean != 0 else 0
            std_dict[metric_key][board_name][model_name] = cv
            max_cv = max(max_cv, cv)
            min_cv = min(min_cv, cv)
            avg_cv += cv
            valid_exps += 1
    
    avg_cv /= valid_exps
    return min_cv, avg_cv, max_cv
            
for metric in [Metrics.LATENCY, Metrics.THROUGHPUT, Metrics.ENERGY]:
    summary_dict = {}
    mean_dict = {}
    std_dict = {}
    metric_key = constants.metric_display_names[metric]
    ga_dict = experiment_utils.get_genetic_algorithm_experiments_all_metrics_boards_models(generations, population,
                                                                                           metric_list=[metric])
    add_dict_values(ga_dict, summary_dict, metric_key)
    for custom_postfix_dir in custom_dir_postfix_dirs_list:
        ga_dict = experiment_utils.get_genetic_algorithm_experiments_all_metrics_boards_models(generations, population,
                                                                                           metric_list=[metric],
                                                                                           custom_dir_postfix = custom_postfix_dir)
        add_dict_values(ga_dict, summary_dict, metric_key)
        
    min_cv, avg_cv, max_cv = mean_stdev(summary_dict, mean_dict, std_dict, metric_key)
    print(min_cv, avg_cv, max_cv)
    out_dir = constants.FIGURES_DATA_DIR_P2 + '/genetic/mean/'
    os.makedirs(constants.FIGURES_DATA_DIR_P2 + out_dir, exist_ok=True)
    mapping_general_utils.save_dict_to_json(mean_dict, out_dir,
                                                metric_key.lower())
    print('------------------------')
    summary_dict = {}
    mean_dict = {}
    std_dict = {}
    sa_dict = experiment_utils.get_simulated_annealing_experiments_all_metrics_boards_models(sa_tenthousands,
                                                                                             metric_list=[metric])
    add_dict_values(sa_dict, summary_dict, metric_key)
    for custom_postfix_dir in custom_dir_postfix_dirs_list:
        sa_dict = experiment_utils.get_simulated_annealing_experiments_all_metrics_boards_models(sa_tenthousands,
                                                                                           metric_list=[metric],
                                                                                           custom_dir_postfix = custom_postfix_dir)
        add_dict_values(sa_dict, summary_dict, metric_key)
        
    min_cv, avg_cv, max_cv = mean_stdev(summary_dict, mean_dict, std_dict, metric_key)
    print(min_cv, avg_cv, max_cv)
    
    out_dir = constants.FIGURES_DATA_DIR_P2 + '/simulated/mean/'
    os.makedirs(constants.FIGURES_DATA_DIR_P2 + out_dir, exist_ok=True)
    mapping_general_utils.save_dict_to_json(mean_dict, out_dir,
                                                metric_key.lower())
    print('------------------------')
    summary_dict = {}
    mean_dict = {}
    std_dict = {}
    sa_dict = experiment_utils.get_hiera_map_algorithm_experiments_all_metrics_boards_models(run_anyway=False,
                                                                                             metric_list=[metric])
    add_dict_values(sa_dict, summary_dict, metric_key)
    for custom_postfix_dir in custom_dir_postfix_dirs_list:
        sa_dict = experiment_utils.get_hiera_map_algorithm_experiments_all_metrics_boards_models(run_anyway=False,
                                                                                           metric_list=[metric],
                                                                                           custom_dir_postfix = custom_postfix_dir)
        add_dict_values(sa_dict, summary_dict, metric_key)
        
    min_cv, avg_cv, max_cv = mean_stdev(summary_dict, mean_dict, std_dict, metric_key)
    print(min_cv, avg_cv, max_cv)
    
    out_dir = constants.FIGURES_DATA_DIR_P2 + '/hiera_map/mean/'
    os.makedirs(constants.FIGURES_DATA_DIR_P2 + out_dir, exist_ok=True)
    mapping_general_utils.save_dict_to_json(mean_dict, out_dir,
                                                metric_key.lower())
    print('------------------------')
        