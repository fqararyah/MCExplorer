import mapping_utils.mapping_exec_utils as mapping_exec_utils
from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
import mapping_utils.custom_mapping_utils as custom_mapping_utils
import optimizers.simulated_annealing as sa
from preformance_record import *
import experiments_p2.experiments as experiments_p2
import experiments_p2.ploting_experiments as plot_exps
from plot import plot_results
from plot import plot_utils
from mapping_types.mapping_description import *
import optimizers.nsga as nsga2
import multiprocessing

num_engines = 10
opt_metric = Metrics.LATENCY

# experiments_p2.find_bests_in_all_base_mappings_metrics_boards_models()

# record, mapping = mapping_exec_utils.run_base_mapping('zc706', 'resnet152', HybridMapping.MAPPING_LABEL, 7)
# print(mapping.calc_exec_time())
# for engine in mapping.get_engines():
#     print(engine.get_parallelism_dims(), engine.num_pes)
# model_dag = utils.read_model_dag_v2(
# constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
# dict = {"0-6": "0-6", "6-155": "6"}
# mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
# board_name, model_dag, dict, timing_metric=Metrics.LATENCY)
# print(mapping.calc_exec_time())
# print(mapping.get_segment_exec_times())
# for sub_mapping in mapping.mapping_list:
#     for engine in sub_mapping.get_engines():
#         print(engine.get_parallelism_dims(), engine.num_pes)


def call_sa():
    # base_dict = experiments_p2.find_bests_in_all_base_mappings_metrics_boards_models()
    # sa_dict = experiments_p2.get_simulated_annealing_experiments_all_metrics_boards_models(
    #     10)
    sa_dict = experiments_p2.get_simulated_annealing_experiments_all_metrics_boards_models(
        20)
    mapping_exec_utils.validate_optimized_mapping_dicts(sa_dict)
    # print('*****************')
    # experiments_p2.validate_base_mapping_dicts(base_dict)
    # #norm_dict = experiments_p2.normalize_performance_dicts(sa_dict, base_dict)
    # best_base_dict = experiments_p2.get_best_base_mappings_in_performance_dicts(base_dict)
    # norm_dict = experiments_p2.normalize_performance_dicts(sa_dict, best_base_dict)

    # print(norm_dict)
    # metric = 'Throughput'
    # plot_dict, board_names, model_names = plot_utils.perf_dict_to_plot_dict(
    #     norm_dict[metric])
    # x_ticks_dict = {len(board_names): mapping_general_utils.get_model_display_name_list(model_names),
    #                 len(model_names): mapping_general_utils.get_board_display_name_list(board_names)}
    # plot_results.plot_bar_groups(plot_dict, 'sa_{}'.format(metric.lower()),
    #                              x_title=None, y_title='SA {} \n reduction'.format(metric),
    #                              x_ticks_dict=x_ticks_dict, relative_save_path='bests',
    #                              abs_save_path=constants.FIGURES_DIR_P2)

# call_ga()
# call_sa_pes()
# call_ga_pes()
# call_sa()


# population = 400
# generations = 50
# sa_tenthousands = 20
# plot_exps.plot_boards_models_metric_vs_all_baselines(constants.metric_display_names[Metrics.LATENCY],
#                                                      population, generations, sa_tenthousands)
# plot_exps.plot_boards_models_metric_vs_all_baselines(constants.metric_display_names[Metrics.THROUGHPUT],
#                                                      population, generations, sa_tenthousands)
# plot_exps.plot_boards_models_metric_vs_best_baseline(constants.metric_display_names[Metrics.LATENCY],
#                                                      population, generations, sa_tenthousands)
# plot_exps.plot_boards_models_metric_vs_best_baseline(constants.metric_display_names[Metrics.THROUGHPUT],
#                                                      population, generations, sa_tenthousands)

# plot_exps.plot_pes_models_metric_vs_best_baseline(constants.metric_display_names[Metrics.THROUGHPUT],
#                                                   population, generations, sa_tenthousands)
# plot_exps.plot_pes_models_metric_vs_best_baseline(constants.metric_display_names[Metrics.LATENCY],
#                                                   population, generations, sa_tenthousands)


population_size = 400
num_generations = 50
hw_file = constants.HW_CONFIGS_FILE

def call_nsga(model_name):

    mapping_exec_utils.run_nsga2_experiments(mapping_general_utils.read_board_names(hw_file),
                                            [model_name], [
                                                Metrics.THROUGHPUT, Metrics.ENERGY],
                                            num_generations, population_size)
    
    mapping_exec_utils.run_nsga2_experiments(mapping_general_utils.read_board_names(hw_file),
                                            [model_name], [
                                                Metrics.LATENCY, Metrics.ENERGY],
                                            num_generations, population_size)

    # mapping_exec_utils.run_nsga2_experiments(mapping_general_utils.read_board_names(hw_file),
    #                                         [model_name], [
    #                                             Metrics.THROUGHPUT, Metrics.BUFFER],
    #                                         num_generations, population_size)

    # mapping_exec_utils.run_nsga2_experiments(mapping_general_utils.read_board_names(hw_file),
    #                                         [model_name], [
    #                                             Metrics.LATENCY, Metrics.BUFFER],
    #                                         num_generations, population_size)

model_names = constants.model_names

processes = []
for model_name in model_names:
    print(model_name)
    processes.append(multiprocessing.Process(target=call_nsga, args=(model_name,)))
    processes[-1].start()

for process in processes:
    process.join()



# # mapping_exec_utils.run_nsga2_experiments([board_name],
# #                                          [model_name], [
# #                                              Metrics.THROUGHPUT, Metrics.BUFFER],
# #                                          50, 100)


# population_size = 400
# generations = 50
# board_name_list = mapping_general_utils.read_board_names(
#     constants.HW_CONFIGS_FILE)
# model_name_list = constants.model_names
# metric1_list = [Metrics.THROUGHPUT]
# metric2_list = [Metrics.BUFFER]

# for metric1 in metric1_list:
#     for metric2 in metric2_list:
#         json_file = ('./figures_data_p2/nsga2/metrics/nsga_{}_{}_population_{}_generations_{}.json'.format(
#             constants.metric_display_names[metric1].lower(),
#             constants.metric_display_names[metric2].lower(),
#             population_size, generations))
#         print(json_file)
#         dict = mapping_general_utils.load_json_to_dict(json_file)
#         for board_name in board_name_list:
#             for model_name in model_name_list:
#                 print(board_name, model_name)
#                 model_dag = utils.read_model_dag_v2(
#                     constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
#                 num_conv_layers = utils.get_num_conv_layer_count_in_range(
#                     model_dag, 0, len(model_dag))

#                 nsga_throughput_list = []
#                 nsga_on_chip_list = []
#                 num_points_dict = {}
#                 for mapping_dict in dict[board_name][model_name]:
#                     mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
#                         board_name, model_dag, mapping_dict, Metrics.THROUGHPUT)
#                     throughput = round(mapping.calc_throughput(), 2)
#                     on_chip = round(
#                         mapping.calc_on_chip_buffer_sz() / constants.MiB, 2)
#                     nsga_throughput_list.append(throughput)
#                     nsga_on_chip_list.append(on_chip)
#                     if throughput not in num_points_dict:
#                         num_points_dict[throughput] = {}
#                     if on_chip not in num_points_dict[throughput]:
#                         num_points_dict[throughput][on_chip] = 0
#                     num_points_dict[throughput][on_chip] += 1

#                 series_labels = []
#                 throughputs_lists = []
#                 on_chip_lists = []
#                 base_mappings_results_dict = mapping_exec_utils.run_base_mappings_engine_range(
#                     board_name, model_name, HybridMapping.MAPPING_LABEL, max_engines=num_conv_layers)

#                 for mapping_label, record_list in base_mappings_results_dict.items():
#                     throughputs_lists.append(mapping_general_utils.get_metric_values_from_perf_records(
#                         record_list, metric1))
#                     on_chip_lists.append(mapping_general_utils.get_metric_values_from_perf_records(
#                         record_list, metric2))
#                     series_labels.append(mapping_label)

#                 series_labels.append('NSGA-II')
#                 throughputs_lists.append(nsga_throughput_list)
#                 on_chip_lists.append(nsga_on_chip_list)

#                 plot_results.scatter(on_chip_lists, throughputs_lists, 'buffers', series_labels,
#                                      ['on-chip buffer (MiB)',
#                                       'Throughput (FPS)'],
#                                      'nsga2_{}_{}_population_{}_generations_{}'.format(board_name, model_name, population_size,
#                                                                                        generations),
#                                      board_name, model_name, figures_path=constants.FIGURES_DIR_P2,
#                                      markers=['x', '*', '^', '.'], alphas=[0.6, 0.6, 0.6, 0.6], figure_size=(3.5, 2))

# # print(optimized_dicts)
