import mapping_utils.mapping_exec_utils as mapping_exec_utils
from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
import mapping_utils.custom_mapping_utils as custom_mapping_utils
from preformance_record import *
from optimizers import hiera_map
from optimizers import simulated_annealing as sa

model_name = 'cmt'
model_dag = utils.read_model_dag_v2(
                constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')

print(model_dag[73])
board_names = ['vcu118']

metric = Metrics.LATENCY
timing_metric = Metrics.THROUGHPUT
if metric == Metrics.LATENCY:
    timing_metric = metric

best_val, best_mapping = mapping_exec_utils.get_best_of_a_mapping(board_names[0], model_name, 
                                         SegmentMappingRR.MAPPING_LABEL, constants.MIN_ENGINES, 
                                         constants.MAX_ENGINES_V2,
                                         metric)

best_mapping_desc_dict = best_mapping.get_dict_representation()
print(best_val, best_mapping_desc_dict)

best_val, best_mapping = mapping_exec_utils.get_best_of_a_mapping(board_names[0], model_name, 
                                         SegmentMapping.MAPPING_LABEL, constants.MIN_ENGINES, 
                                         constants.MAX_ENGINES_V2,
                                         metric)

best_mapping_desc_dict = best_mapping.get_dict_representation()
print(best_val, best_mapping_desc_dict)

best_val, best_mapping = mapping_exec_utils.get_best_of_a_mapping(board_names[0], model_name, 
                                         HybridMapping.MAPPING_LABEL, constants.MIN_ENGINES, 
                                         constants.MAX_ENGINES_V2,
                                         metric)

best_mapping_desc_dict = best_mapping.get_dict_representation()
print(best_val, best_mapping_desc_dict)

mapping, mapping_desc = custom_mapping_utils.generate_random_mapping(
                        board_names[0], model_name, model_dag, 2, timing_metric=timing_metric)

best_val, best_mapping = hiera_map.run_hiera_map(mapping_desc, metric, constants.MAX_CLUSTERS)
print(best_val, best_mapping)


best_val, best_mapping = sa.run_simulated_annealing(mapping, mapping_desc, metric, 20000)
mapping_desc_dict = custom_mapping_utils.prepare_custom_mapping_desc(
                    best_mapping.segment_layers_list,
                    best_mapping.segment_block_list,
                    best_mapping.block_engines_list)

best_sa_mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
                   board_names[0], model_dag, mapping_desc_dict, timing_metric, adjust_pes=best_mapping.adjust_pes)

perf_record = mapping_exec_utils.run_mapping(board_names[0], model_name, best_sa_mapping)
best_val = perf_record.get_metric_val(metric)

print(best_val, mapping_desc_dict)