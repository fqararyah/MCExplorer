import __init__
import mapping_utils.custom_mapping_utils as custom_mapping_utils
import mapping_utils.mapping_exec_utils as mapping_exec_utils
import mapping_utils.mapping_general_utils as mapping_general_utils
from preformance_record import *
import utils
import constants
from plot import plot_results
import random
from plot import plot_utils

if not constants.TEST_SEEDS:
    random.seed(5)
    
num_archs = 200
board_name = 'zcu102'
model_name = 'resnet50'
model_dag = utils.read_model_dag_v2(
    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')

throughputs = []
energies = []
mapping_dicts = []
for i in range(num_archs):
    mapping, mapping_desc =\
        custom_mapping_utils.generate_random_mapping(
            board_name, model_name, model_dag, timing_metric=Metrics.THROUGHPUT,
            max_engines_bounded_by_layers=True)
    mapping_dict = custom_mapping_utils.prepare_custom_mapping_desc(
        mapping_desc.segment_layers_list,
        mapping_desc.segment_block_list,
        mapping_desc.block_engines_list)
    mapping_dicts.append(mapping)
    perf_rec = mapping_exec_utils.run_mapping(board_name, model_name, mapping)
    throughputs.append(perf_rec.throughput)
    energies.append(perf_rec.energy)

min_throughput = min(throughputs)
min_energy = min(energies)
print(mapping_dicts[throughputs.index(min_throughput)], min_throughput)
print(max(throughputs) / min_throughput)
print(max(energies) / min_energy)
norm_throughputs = [throughput / min_throughput for throughput in throughputs]
norm_energies = [energy / min_energy for energy in energies]
x_points = [*range(1, num_archs + 1)]

save_path = mapping_general_utils.prepare_save_path(
    constants.FIGURES_DIR_P2, '/random_{}/'.format(num_archs))

plot_results.scatter_one_series(x_points, norm_throughputs, ['accelerator', 'normalized throughput'],
                                'random_archs_throughput', save_path,
                                figure_size=plot_utils.proportional_fig_size(x_points, norm_throughputs,
                                                                             plot_utils.HALF_PAGE, height_multiplier=0.5,
                                                                             proportional_to_points=False),
                                set_xlimit_to_data=True,
                                gradient=True)

plot_results.scatter_one_series(x_points, norm_energies, ['accelerator', 'normalized energy'],
                                'random_archs_energy', save_path,
                                figure_size=plot_utils.proportional_fig_size(x_points, norm_energies,
                                                                             plot_utils.HALF_PAGE, height_multiplier=0.5,
                                                                             proportional_to_points=False),
                                set_xlimit_to_data=True,
                                gradient=True, higher_is_better=False)
