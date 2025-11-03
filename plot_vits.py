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
from mapping_types.mapping_description import *
import optimizers.nsga as nsga2
import multiprocessing
from optimizers import hiera_map
import matplotlib.pyplot as plt

def call_sa(metric):
    sa_dict = experiment_utils.get_simulated_annealing_experiments_all_metrics_boards_models(
        20000, [metric], run_anyway=False)
    mapping_exec_utils.validate_optimized_mapping_dicts(sa_dict)


def call_ga(metric, population=400):
    ga_dict = experiment_utils.get_genetic_algorithm_experiments_all_metrics_boards_models(
        50, population, [metric], run_anyway=False)

def call_hiera_map(metric):
    model_name_list = ['cmt', 'unet']
    hm_dict = experiment_utils.get_hiera_map_algorithm_experiments_all_metrics_boards_models(run_anyway=False,
                                                                                             metric_list=[metric],
                                                                                             model_name_list=model_name_list,
                                                                                             custom_dir_postfix = 'vits')


population = 400
generations = 50
sa_tenthousands = 20000
hm_max_clusters = constants.MAX_CLUSTERS
run_base_anyway = False

plt.figure(figsize=(8.5, 1))
print('VVV')
plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.THROUGHPUT,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=False,
                                                     plot_average=False,
                                                     plot_legend = False,
                                                     min_y_lim = 0,
                                                     sub_plot_params = (1, 3, 1),
                                                     custom_dir_postfix = 'vits')
print('VVV')

plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.LATENCY,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=False,
                                                     plot_average=False,
                                                     plot_legend = False,
                                                     min_y_lim = 0,
                                                     sub_plot_params = (1, 3, 2),
                                                     custom_dir_postfix = 'vits')
print('VVV')

plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.ENERGY,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=False,
                                                     plot_average=False,
                                                     plot_legend = False,
                                                     min_y_lim = 0,
                                                     sub_plot_params = (1, 3, 3),
                                                     custom_dir_postfix = 'vits')

#plt.tight_layout()
plt.subplots_adjust(wspace=0.3)

plt.savefig('./boards_normalized_all_metrics_best_vits.png', format='png',
                    bbox_inches='tight')
plt.savefig('./boards_normalized_all_metrics_best_vits.pdf', format='pdf',
            bbox_inches='tight')
plt.clf()
plt.close()