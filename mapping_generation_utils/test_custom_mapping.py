from mapping_types.generic_mapping import *
from mapping_types.sesl_mapping import *
from mapping_strategies.mapping_types.seml_mapping_lbl import *
from mapping_strategies.mapping_types.seml_mapping_fused import *
from mapping_strategies.mapping_types.segment_grained_mapping_rr import *
from mapping_strategies.mapping_types.segment_grained_mapping import *
from mapping_strategies.mapping_types.hybrid_mapping import *
from mapping_strategies.mapping_types.hybrid_rr_mapping import *
from mapping_strategies.mapping_utils.custom_mapping_utils import *
import __init__
import mapping_strategies.mapping_utils.mapping_general_utils as mapping_general_utils
from hw_config import *
import experiments.experiments as exps exps
from preformance_record import *
from mapping_strategies.mapping_types.custom_mapping import *
import os


for json_mapping_dir in os.listdir(constants.CUSTOM_MAPPINGS_JSON_DIR):
    file_path = constants.CUSTOM_MAPPINGS_JSON_DIR + '/' + '/' + json_mapping_dir + '/'
    errors_list = []
    for json_mapping_file in os.listdir(file_path):
        if not '.json' in json_mapping_file:
            continue
        mapping_configs_list = mapping_general_utils.read_mappings_json(
            file_path + json_mapping_file)
        for i in range(len(mapping_configs_list)):
            current_mapping_configs = mapping_configs_list[i]

            model_name = current_mapping_configs['model_name']
            board_name = current_mapping_configs['board_name']
            model_dag = utils.read_model_dag_v2(
                constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')

            layers = utils.get_conv_layer_indices_in_range(
                model_dag, 0, len(model_dag))

            mapping_details = current_mapping_configs['layer_engine_mapping']

            mappings_segments_config_list = infer_mapping_types(
                mapping_details, model_dag)
            hw_config = HWConfig(board_name)

            if json_mapping_dir == HybridMapping.MAPPING_LABEL:
                mapping1 = HybridMapping(hw_config, model_dag, layers, len(mappings_segments_config_list[0]['engine_list']),
                                         first_layer_ifms_are_on_chip=False,
                                         last_layer_ofms_are_on_chip=False)
            elif json_mapping_dir == SegmentMappingRR.MAPPING_LABEL:
                mapping1 = SegmentMappingRR(hw_config, model_dag, layers, len(
                    mappings_segments_config_list[0]['engine_list']))
            elif json_mapping_dir == SegmentMapping.MAPPING_LABEL:
                mapping1 = SegmentMapping(
                    hw_config, model_dag, layers, len(mappings_segments_config_list))

            mapping2 = CustomMapping(hw_config, model_dag, layers, mappings_segments_config_list,
                                     first_layer_ifms_are_on_chip=False,
                                     last_layer_ofms_are_on_chip=False,
                                     apply_fusion=False)

            exec_time_diff = mapping1.calc_exec_time() - mapping2.calc_exec_time()
            fms_buffer_sz_diff = mapping1.calc_fms_buffer_sz() - mapping2.calc_fms_buffer_sz()
            weights_buffer_sz_diff = mapping1.calc_weights_buffer_sz() - \
                mapping2.calc_weights_buffer_sz()
            weights_access_diff = mapping1.calc_off_chip_weights_access() - \
                mapping2.calc_off_chip_weights_access()
            fms_access_diff = mapping1.calc_off_chip_fms_access() - \
                mapping2.calc_off_chip_fms_access()

            if exec_time_diff != 0 or fms_buffer_sz_diff != 0 or weights_buffer_sz_diff != 0 or \
                    weights_access_diff != 0 or fms_access_diff != 0:
                errors_list.append('{} {} {}'.format(
                    board_name, model_name, mapping_details))
            if exec_time_diff != 0:
                errors_list.append('exec_time_diff: {}, as a percentage: {}'.format(
                    exec_time_diff,
                    abs(exec_time_diff) / mapping1.calc_exec_time()))
            if fms_buffer_sz_diff != 0:
                errors_list.append(
                    'fms_buffer_sz_diff: {}'.format(fms_buffer_sz_diff / constants.MiB))
            if weights_buffer_sz_diff != 0:
                errors_list.append(
                    'weights_buffer_sz_diff: {}'.format(weights_buffer_sz_diff / constants.MiB))
            if weights_access_diff != 0:
                errors_list.append(
                    'weights_access_diff: {}'.format(weights_access_diff / constants.MiB))
            if fms_access_diff != 0:
                errors_list.append(
                    'fms_access_diff: {}'.format(fms_access_diff / constants.MiB))

    with open(file_path + 'specialization_diffs.txt', 'w') as f:
        for line in errors_list:
            f.write(line + '\n')
