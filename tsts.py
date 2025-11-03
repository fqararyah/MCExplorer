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

def call_sa(metric):
    sa_dict = experiment_utils.get_simulated_annealing_experiments_all_metrics_boards_models(
        20000, [metric], run_anyway=False)
    mapping_exec_utils.validate_optimized_mapping_dicts(sa_dict)


def call_ga(metric, population=400):
    ga_dict = experiment_utils.get_genetic_algorithm_experiments_all_metrics_boards_models(
        50, population, [metric], run_anyway=False)

def call_hiera_map(metric):
    hm_dict = experiment_utils.get_hiera_map_algorithm_experiments_all_metrics_boards_models(run_anyway=False,
                                                                                             metric_list=[metric])


population = 400
generations = 50
sa_tenthousands = 20000
hm_max_clusters = constants.MAX_CLUSTERS
run_base_anyway = False

call_hiera_map(Metrics.THROUGHPUT)
plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.THROUGHPUT,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=True,
                                                     plot_average=False,
                                                     min_y_lim = 1)
plot_exps.plot_boards_models_metric_vs_all_baselines(Metrics.THROUGHPUT,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=True,
                                                     plot_average=True,
                                                     min_y_lim = 1)

plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.THROUGHPUT,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, 
                                                     plot_ideal=False,
                                                     black_box=False,
                                                     plot_average=False,
                                                     custom_dir_postfix='mean',
                                                     file_name='throughput.json')
call_hiera_map(Metrics.LATENCY)
plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.LATENCY,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=True,
                                                     plot_average=False,
                                                     min_y_lim = 1)
plot_exps.plot_boards_models_metric_vs_all_baselines(Metrics.LATENCY,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=True,
                                                     plot_average=True,
                                                     min_y_lim = 1)

plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.LATENCY,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, 
                                                     plot_ideal=False,
                                                     black_box=False,
                                                     plot_average=False,
                                                     custom_dir_postfix='mean',
                                                     file_name='latency.json')

call_hiera_map(Metrics.ENERGY)
plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.ENERGY,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=False,
                                                     saving=True,
                                                     plot_average=True)
plot_exps.plot_boards_models_metric_vs_all_baselines(Metrics.LATENCY,
                                                     population, generations, sa_tenthousands, hm_max_clusters,
                                                     run_base_anyway=run_base_anyway, plot_ideal=True,
                                                     plot_average=True,
                                                     min_y_lim = 1)

plot_exps.plot_boards_models_metric_vs_best_baseline(Metrics.ENERGY,
                                                     population, generations, sa_tenthousands, hm_max_clusters, 
                                                     black_box=False,
                                                     plot_average=False,
                                                     custom_dir_postfix='mean',
                                                     file_name='energy.json'
                                                     #saving=True
                                                     )

def plot_things():
    population_size = 400
    generations = 50
    board_name_list = ['zc706']#mapping_general_utils.read_board_names(constants.HW_CONFIGS_FILE)
    model_name_list = ['dense121']#constants.model_names
    metric1_list = [Metrics.LATENCY, Metrics.THROUGHPUT]
    metric2_list = [Metrics.ENERGY]
    y_axis_strs = {
        Metrics.LATENCY: 'Latency (ms)', Metrics.THROUGHPUT: 'Throughput (FPS)'}
    x_axis_strs = {
        Metrics.ENERGY: 'Energy (mJ / inference)'}

    base_mappings_results_dict = experiment_utils.get_base_mappings_performance_all_boards_models(
        max_engines_bounded_by_layers=True, run_anyway=False)

    for metric1 in metric1_list:
        for metric2 in metric2_list:
            for board_name in board_name_list:
                for model_name in model_name_list:
                    print('plot_things >> ', board_name, model_name)
                    model_dag = utils.read_model_dag_v2(
                        constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                    json_file = ('./figures_data_p2/nsga2/models/nsga_{}_{}_{}_{}_population_{}_generations_{}.json'.format(
                        constants.metric_display_names[metric1].lower(),
                        constants.metric_display_names[metric2].lower(),
                        board_name,
                        model_name,
                        population_size, generations))
                    dict = mapping_general_utils.load_json_to_dict(json_file)

                    nsga_y_list = []
                    nsga_x_list = []
                    for mapping_dict in dict[model_name]:
                        timig_metric = Metrics.THROUGHPUT
                        if metric1 == Metrics.LATENCY:
                            timig_metric = Metrics.LATENCY
                        mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
                            board_name, model_dag, mapping_dict, timig_metric)

                        perf_record = mapping_exec_utils.run_mapping(
                            board_name, model_name, mapping)
                        x_val = perf_record.get_metric_val(metric2)
                        y_val = perf_record.get_metric_val(metric1)
                        on_chip = round(
                            mapping.calc_on_chip_buffer_sz() / constants.MiB, 2)
                        nsga_y_list.append(y_val)
                        nsga_x_list.append(x_val)

                    series_labels = []
                    y_lists = []
                    x_lists = []

                    for mapping_label, record_list in base_mappings_results_dict[board_name][model_name].items():
                        y_lists.append(mapping_general_utils.get_metric_values_from_perf_records_dicts(
                            record_list, metric1))

                        x_lists.append(mapping_general_utils.get_metric_values_from_perf_records_dicts(
                            record_list, metric2))

                        series_labels.append(mapping_label)

                    series_labels.append('NSGA-II')
                    y_lists.append(nsga_y_list)
                    x_lists.append(nsga_x_list)

                    plot_results.scatter_with_zoom(x_lists, y_lists,
                                                     constants.metric_file_names[metric1] +
                                                     '_' +
                                                     constants.metric_file_names[metric2],
                                                     series_labels, [
                                                         x_axis_strs[metric2], y_axis_strs[metric1]],
                                                     'nsga2_{}_{}_{}_{}_population_{}_generations_{}'.format(board_name, model_name,
                                                                                                             constants.metric_display_names[
                                                                                                                 metric1],
                                                                                                             constants.metric_display_names[
                                                                                                                 metric2],
                                                                                                             population_size,
                                                                                                             generations),
                                                     board_name, model_name, figures_path=constants.FIGURES_DIR_P2,
                                                     markers=['x', '*', '^', '.'],
                                                     alphas=[0.6, 0.6, 0.6, 0.6], figure_size=(3.5, 1.5),
                                                     scale_x = 1000)


#plot_things()

#processes = []

# processes.append(multiprocessing.Process(target=call_ga, args=(Metrics.LATENCY, population, )))
# processes.append(multiprocessing.Process(target=call_sa, args=(Metrics.LATENCY, )))
#call_ga(Metrics.LATENCY, population)
#call_sa(Metrics.LATENCY)


# processes.append(multiprocessing.Process(target=call_ga, args=(Metrics.THROUGHPUT, population, )))
# processes.append(multiprocessing.Process(target=call_sa, args=(Metrics.THROUGHPUT, )))
#call_ga(Metrics.THROUGHPUT, population)
#call_sa(Metrics.THROUGHPUT)

# processes.append(multiprocessing.Process(target=call_ga, args=(Metrics.ENERGY, population, )))
# processes.append(multiprocessing.Process(target=call_sa, args=(Metrics.ENERGY, )))
#call_ga(Metrics.ENERGY, population)
#call_sa(Metrics.ENERGY)

# for process in processes:
#     process.start()

# for process in processes:
#     process.join()