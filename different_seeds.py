
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


def call_sa(metric, seed_val):
    #board_name_list = ['zc706']#mapping_general_utils.read_board_names(constants.HW_CONFIGS_FILE)
    model_name_list = ['resnet152', 'resnet50', 'xce_r', 'dense121', 'mob_v2']#constants.model_names
    sa_dict = experiment_utils.get_simulated_annealing_experiments_all_metrics_boards_models(
        20000, [metric], run_anyway=True, custom_dir_postfix=str(seed_val), model_name_list=model_name_list)


def call_ga(metric, seed_val):
    #board_name_list = ['zc706']#mapping_general_utils.read_board_names(constants.HW_CONFIGS_FILE)
    model_name_list = ['resnet152', 'resnet50', 'xce_r', 'dense121', 'mob_v2']#constants.model_names
    ga_dict = experiment_utils.get_genetic_algorithm_experiments_all_metrics_boards_models(
        50, metric_list= [metric], population_size=400,
        run_anyway=True, custom_dir_postfix=str(seed_val), model_name_list=model_name_list)

def call_hiera_map(metric, seed_val):
    hm_dict = experiment_utils.get_hiera_map_algorithm_experiments_all_metrics_boards_models(run_anyway=True,
                                                                                             metric_list=[metric], 
                                                                                             custom_dir_postfix=seed_val)
          
#for i in range(5):
#    print('iterrrrrr: {}'.format(i))
seed = 30
random.seed(seed)
call_sa(Metrics.LATENCY, seed)
call_ga(Metrics.LATENCY, seed)
call_ga(Metrics.THROUGHPUT, seed)
call_ga(Metrics.THROUGHPUT, seed)
call_ga(Metrics.ENERGY, seed)
call_ga(Metrics.ENERGY, seed)
# call_hiera_map(Metrics.THROUGHPUT, seed)
# call_hiera_map(Metrics.ENERGY, seed)
#call_hiera_map(Metrics.LATENCY, seed)