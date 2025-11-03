
from generic_mapping import GenericMapping
import __init__
import utils
from mapping_strategies.engines.engine import *
from seml_mapping_lbl import *
from seml_mapping_fused import *
from sesl_mapping import *
from segment_grained_mapping_rr import *
from hw_config import *
import copy
from preformance_record import Metrics
from mapping_utils import custom_mapping_utils
from basic_mapping import *


class CustomMapping(BasicMapping):
    DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS = 1
    MAPPING_LABEL = 'Custom'

    def __init__(self, hw_config, model_dag, layers, mappings_segments_config_list,
                 rows_to_produce_in_pipe_pass=DEFAULT_ROWS_TO_PRODUCE_IN_A_PASS,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 inter_segment_layers_on_chip = True,
                 apply_fusion=False,
                 timing_metric=Metrics.THROUGHPUT,
                 adjust_pes=False,
                 adjusted_pes_list=None, 
                 enable_multi_ce = True):

        super().__init__(hw_config, model_dag, layers, [])
        self.rows_to_produce_in_pipe_pass = rows_to_produce_in_pipe_pass
        self.num_layers = len(layers)
        self.first_layer_ifms_are_on_chip = first_layer_ifms_are_on_chip
        self.last_layer_ofms_are_on_chip = last_layer_ofms_are_on_chip
        self.apply_fusion = apply_fusion
        self.mappings_segments_config_list = mappings_segments_config_list
        self.timing_metric = timing_metric
        self.num_segments = len(mappings_segments_config_list)
        self.mapping_list = []
        self.adjust_pes = adjust_pes
        self.adjusted_pes_list = adjusted_pes_list
        self.inter_segment_layers_on_chip = inter_segment_layers_on_chip
        self.enable_multi_ce = enable_multi_ce
        self.initialize_mappings()

    @log_function_call
    def balance_pes(self):
        overall_op_count = sum(utils.get_layers_op_counts(self.model_dag))
        segment_op_counts_list = []
        for _, layer_set in self.mappings_layers_dict.items():
            self.mappings_hw_configs.append(self.hw_config.copy_hw_config())

            current_op_count = sum(
                utils.get_layers_op_counts_by_indices(self.model_dag, layer_set))
            segment_op_counts_list.append(current_op_count)

            if self.timing_metric == Metrics.THROUGHPUT:
                current_pes = max(1, (current_op_count *
                                      self.hw_config.num_pes) // overall_op_count)
                self.mappings_hw_configs[-1].num_pes = current_pes

        if self.timing_metric == Metrics.LATENCY:
            engine_pes_list = custom_mapping_utils.proportional_allocation(
                self.hw_config.num_pes, segment_op_counts_list)
            for i in range(len(self.mappings_hw_configs)):
                self.mappings_hw_configs[i].num_pes = engine_pes_list[i]
            # print(engine_pes_list)

    def __str__(self):
        return str(self.get_dict_representation())
    
    @log_function_call
    def initialize_tmp_mappings(self):
        tmp_mappings = [None] * self.num_segments
        for i in range(0, 2):
            mappings_exec_times = []
            mapping_indices = []
            for segment_index, config_dict in enumerate(self.mappings_segments_config_list):
                first_layer_ifms_are_on_chip = (segment_index > 0 and self.inter_segment_layers_on_chip) or \
                      self.first_layer_ifms_are_on_chip
                last_layer_ofms_are_on_chip = (segment_index < self.num_segments - \
                    1 and self.inter_segment_layers_on_chip) or self.last_layer_ofms_are_on_chip
                mapping_label = config_dict['mapping']
                engine_list = config_dict['engine_list']
                layer_list = config_dict['layer_list']
                tmp_mapping = None
                
                if mapping_label == SESLMapping.MAPPING_LABEL:
                    tmp_mapping = SESLMapping(self.mappings_hw_configs[segment_index],
                                                self.model_dag, layer_list,
                                                first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                engine_parallelization_strategy=ParallelizationStrategies.CUSTOM,
                                                timing_metric=self.timing_metric)
                elif mapping_label == SegmentMappingRR.MAPPING_LABEL:
                    tmp_mapping = SegmentMappingRR(
                        self.mappings_hw_configs[segment_index], self.model_dag, layer_list, len(
                            engine_list),
                        first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                        last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                        engine_parallelization_strategy=ParallelizationStrategies.CUSTOM)
                elif mapping_label == SEMLMapping_LBL.MAPPING_LABEL:
                    tmp_mapping = SEMLMapping_LBL(self.mappings_hw_configs[segment_index],
                                                    self.model_dag, layer_list,
                                                    first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                    last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                    dynamic_fm_placement_policy = True,
                                                    pw_conv_parallelization_strategy=ParallelizationStrategies.CUSTOM,
                                                    adjust_pes=True,
                                                    enable_multi_ce=self.enable_multi_ce)
                tmp_mappings[segment_index] = tmp_mapping

                if i == 0:
                    mappings_exec_times.append(
                        tmp_mappings[segment_index].calc_exec_time())
                    mapping_indices.append(segment_index)
            
            if self.adjust_pes and i == 0:
                zipped_lists = zip(mappings_exec_times, mapping_indices)
                zipped_lists = sorted(zipped_lists, reverse=True)
                _, sorted_mapping_indices = zip(*zipped_lists)
                unused_pes = self.hw_config.num_pes
                segment_index = 0
                segment_used_pes_list = []
                segments_used_pe_sums = []
                for mapping in tmp_mappings:
                    segment_used_pes = 0
                    segment_used_pes_list.append([])
                    for eng in mapping.get_engines():
                        segment_used_pes += eng.get_parallelism()
                        segment_used_pes_list[-1].append(eng.get_parallelism())
                    segments_used_pe_sums.append(segment_used_pes)
                    unused_pes -= segment_used_pes
                    self.mappings_hw_configs[segment_index].num_pes = segment_used_pes
                    segment_index += 1

                # segment_index = 0
                # print(mappings_exec_times)
                # for mapping in tmp_mappings:
                #     print(self.mappings_hw_configs[segment_index].num_pes)
                #     segment_index += 1
                #     for eng in mapping.get_engines():
                #         print(eng.get_parallelism_dims())
                #     print('*****************')
                # print(unused_pes)

                for segment_index in sorted_mapping_indices:
                    num_engines = len(segment_used_pes_list[segment_index])
                    for engine_index in range(num_engines):
                        engine_pes = segment_used_pes_list[segment_index][engine_index]
                        if unused_pes >= engine_pes:
                            self.mappings_hw_configs[segment_index].num_pes += engine_pes
                            unused_pes -= engine_pes

                if unused_pes > 0:
                    all_used_pes = 0
                    for segment_index in range(self.num_segments):
                        all_used_pes += self.mappings_hw_configs[segment_index].num_pes
                    unused_pes = self.hw_config.num_pes - all_used_pes
                    extra_pes_by_segments = unused_pes // self.num_segments
                    reminder_pes = unused_pes % self.num_segments
                    for segment_index in sorted_mapping_indices:
                        self.mappings_hw_configs[segment_index].num_pes += extra_pes_by_segments
                        if reminder_pes > 0:
                            self.mappings_hw_configs[segment_index].num_pes += 1
                            reminder_pes -= 1
                
                # print('AFTER')
                # segment_index = 0
                # for mapping in tmp_mappings:
                #     print(self.mappings_hw_configs[segment_index].num_pes)
                #     segment_index += 1
                #     for eng in mapping.get_engines():
                #         print(eng.get_parallelism_dims())
                #     print('*****************')

        return tmp_mappings

    @log_function_call
    def distribute_on_chip_memory(self, tmp_mappings):
        segment_buffers = []
        for _, segments_index_list in self.first_engine_segment_index_dict.items():
            current_segment_max_buffer_sz = 0
            for segment_index in segments_index_list:
                current_segment_max_buffer_sz = max(current_segment_max_buffer_sz,
                                                    tmp_mappings[segment_index].calc_on_chip_buffer_sz_pure())
            segment_buffers.append(current_segment_max_buffer_sz)

        total_segment_buffers = sum(segment_buffers)
        for i in range(len(segment_buffers)):
            self.mappings_hw_configs[i].on_chip_memory = max((
                segment_buffers[i] * self.hw_config.on_chip_memory) // total_segment_buffers, 1)

    @log_function_call
    def initialize_mappings(self):
        self.segment_first_engine_dict = {}
        self.segment_engine_set_list = []
        self.segment_layer_set_list = []
        self.first_engine_segment_index_dict = {}
        self.mappings_layers_dict = {}
        self.mappings_hw_configs = []
        self.mappings = []

        segment_index = 0
        for config_dict in self.mappings_segments_config_list:
            engine_list = config_dict['engine_list']
            layer_list = config_dict['layer_list']
            first_engine_id = engine_list[0]
            self.segment_first_engine_dict[segment_index] = first_engine_id
            self.segment_engine_set_list.append(engine_list)
            self.segment_layer_set_list.append(layer_list)
            if first_engine_id not in self.first_engine_segment_index_dict:
                self.first_engine_segment_index_dict[first_engine_id] = []
                self.mappings_layers_dict[first_engine_id] = []
            self.first_engine_segment_index_dict[first_engine_id].append(
                segment_index)
            self.mappings_layers_dict[first_engine_id].extend(layer_list)

            segment_index += 1

        if self.adjusted_pes_list is None:
            self.balance_pes()
        else:
            for i in range(self.num_segments):
                self.mappings_hw_configs.append(
                    self.hw_config.copy_hw_config())
                self.mappings_hw_configs[-1].num_pes = self.adjusted_pes_list[i]

        tmp_mappings = self.initialize_tmp_mappings()
        self.distribute_on_chip_memory(tmp_mappings)

        segment_index = 0
        for config_dict in self.mappings_segments_config_list:
            first_layer_ifms_are_on_chip = (segment_index > 0 and self.inter_segment_layers_on_chip) or \
                      self.first_layer_ifms_are_on_chip
            last_layer_ofms_are_on_chip = (segment_index < self.num_segments - \
                1 and self.inter_segment_layers_on_chip) or self.last_layer_ofms_are_on_chip
            mapping_label = config_dict['mapping']
            engine_list = config_dict['engine_list']
            layer_list = config_dict['layer_list']
            if mapping_label == SESLMapping.MAPPING_LABEL:
                self.mapping_list.append(SESLMapping(self.mappings_hw_configs[segment_index],
                                                     self.model_dag, layer_list,
                                                     engines=tmp_mappings[segment_index].engines,
                                                     pre_balanced_engines=True,
                                                     first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                     last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                     engine_parallelization_strategy=ParallelizationStrategies.CUSTOM,
                                                     timing_metric=self.timing_metric))
            elif mapping_label == SegmentMappingRR.MAPPING_LABEL:
                self.mapping_list.append(SegmentMappingRR(
                    self.mappings_hw_configs[segment_index], self.model_dag, layer_list, len(
                        engine_list),
                    engines=tmp_mappings[segment_index].engines,
                    pre_balanced_engines=True,
                    first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                    last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                    engine_parallelization_strategy=ParallelizationStrategies.CUSTOM))
            elif mapping_label == SEMLMapping_LBL.MAPPING_LABEL:
                self.mapping_list.append(SEMLMapping_LBL(self.mappings_hw_configs[segment_index],
                                                         self.model_dag, layer_list,
                                                         engines=tmp_mappings[segment_index].engines,
                                                         pre_balanced_engines=True,
                                                         first_layer_ifms_are_on_chip=first_layer_ifms_are_on_chip,
                                                         last_layer_ofms_are_on_chip=last_layer_ofms_are_on_chip,
                                                         dynamic_fm_placement_policy = True,
                                                         pw_conv_parallelization_strategy=ParallelizationStrategies.CUSTOM,
                                                         adjust_pes=True,
                                                         enable_multi_ce=self.enable_multi_ce))
            segment_index += 1

    @log_function_call
    def get_label(self):
        return self.MAPPING_LABEL

    @log_function_call
    def calc_exec_time(self, print_desc=False):
        exec_time = 0
        desc_str = ' '
        segment_exec_times = self.get_segment_exec_times()
        already_calculated = {}
        for segment_index, mapping in enumerate(self.mapping_list):
            if segment_index in already_calculated:
                continue
            if segment_index + 1 < self.num_segments and (isinstance(mapping, SESLMapping) or len(mapping.layers) == 1) and \
                    isinstance(self.mapping_list[segment_index + 1], SESLMapping):
                sesl_mapping = self.mapping_list[segment_index + 1]
                sesl_pipe_passes = sesl_mapping.get_pipe_num_passes()
                current_mapping_pass_time = segment_exec_times[segment_index] / \
                    sesl_pipe_passes
                sels_mapping_pass_time = segment_exec_times[segment_index + 1] / \
                    sesl_pipe_passes
                combined_time = current_mapping_pass_time + \
                    max(current_mapping_pass_time,
                        sels_mapping_pass_time) * sesl_pipe_passes

                exec_time += combined_time
                already_calculated[segment_index + 1] = 1
            else:
                exec_time += segment_exec_times[segment_index]
                desc_str += str(exec_time) + ' '

        return exec_time

    def get_segment_exec_times(self):
        exec_times = []
        for mapping in self.mapping_list:
            exec_times.append(mapping.calc_exec_time())

        return exec_times

    def get_num_engines(self):
        all_engines = 0
        for mapping in self.mapping_list:
            all_engines += (mapping.get_num_engines())

        return int(all_engines)

    @log_function_call
    def calc_throughput(self):
        max_exec_time = 0
        for mapping in self.mapping_list:
            max_exec_time = max(max_exec_time, mapping.calc_exec_time())

        return 1 / max_exec_time

    @log_function_call
    def calc_fms_buffer_sz(self, print_desc=False):
        fms_buffer_sz = 0
        if print_desc:
            print(self.MAPPING_LABEL)
        for _, segments_index_list in self.first_engine_segment_index_dict.items():
            current_segment_max_fms_buffer_sz = 0
            for segment_index in segments_index_list:
                current_segment_max_fms_buffer_sz = max(current_segment_max_fms_buffer_sz,
                                                        self.mapping_list[segment_index].calc_fms_buffer_sz(print_desc))

            fms_buffer_sz += current_segment_max_fms_buffer_sz
            if print_desc:
                print(fms_buffer_sz)

        return fms_buffer_sz

    @log_function_call
    def calc_weights_buffer_sz(self):
        weights_buffer_sz = 0
        for _, segments_index_list in self.first_engine_segment_index_dict.items():
            current_segment_max_weights_buffer_sz = 0
            for segment_index in segments_index_list:
                current_segment_max_weights_buffer_sz = max(current_segment_max_weights_buffer_sz,
                                                            self.mapping_list[segment_index].calc_weights_buffer_sz())

            weights_buffer_sz += current_segment_max_weights_buffer_sz

        return weights_buffer_sz
    
    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    @log_function_call
    def calc_off_chip_weights_access(self):
        off_chip_weight_accesses = 0
        for mapping in self.mapping_list:
            off_chip_weight_accesses += mapping.calc_off_chip_weights_access()

        return off_chip_weight_accesses

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    @log_function_call
    def calc_off_chip_fms_access(self, print_desc=False):
        off_chip_fms_accesses = 0
        desc_str = ''
        for mapping in self.mapping_list:
            segment_off_chip_access = mapping.calc_off_chip_fms_access()
            off_chip_fms_accesses += segment_off_chip_access
            desc_str += '{} '.format(segment_off_chip_access)

        if print_desc:
            print(self.MAPPING_LABEL, 'calc_off_chip_fms_access', desc_str)

        return off_chip_fms_accesses

    @log_function_call
    def get_engines(self):
        engines = []
        for mapping in self.mapping_list:
            engines.extend(mapping.engines)

        return engines

    @log_function_call
    def get_off_chip_tmp_channels_layers(self):
        are_tmp_channels_on_chip = []
        for mapping in self.mapping_list:
            are_tmp_channels_on_chip.extend(
                mapping.get_off_chip_tmp_channels_layers())

    @log_function_call
    def get_dict_representation(self):
        mapping_dict = {}
        starting_engine = 0
        starting_layer = 0
        for mapping in self.mapping_list:
            num_engines = mapping.get_num_engines()
            num_layers = mapping.get_num_layers()
            layers_str = str(starting_layer)
            engines_str = str(starting_engine)
            if num_layers > 1:
                layers_str += '-{}'.format(starting_layer + num_layers)
            if num_engines > 1:
                engines_str += '-{}'.format(starting_engine + num_engines)

            starting_engine += num_engines
            starting_layer += num_layers

            mapping_dict[layers_str] = engines_str

        return mapping_dict

    @log_function_call
    def calc_energy(self, print_desc = False):
        exec_time = self.calc_exec_time()
        bram_energy_ifms_idle = 0
        bram_energy_ofms_idle = 0
        bram_energy_weights_idle = 0
        bram_energy_ifms_accesses = 0
        bram_energy_ofms_accesses = 0
        bram_energy_weights_accesses = 0
        off_chip_accesses_energy = 0
        dsps_idle_energy = 0
        dsps_comp_energy = 0
        
        for layer_group_index in range(len(self.mapping_list)):
            energy_comps = self.mapping_list[layer_group_index].calc_energy_components()

            bram_energy_ifms_idle += energy_comps.on_chip_ifms_idle_energy_per_second * exec_time
            bram_energy_ofms_idle += energy_comps.on_chip_ofms_idle_energy_per_second * exec_time
            bram_energy_weights_idle += energy_comps.on_chip_weights_idle_energy_per_second * exec_time

            bram_energy_ifms_accesses += energy_comps.on_chip_ifms_dynamic_energy
            bram_energy_ofms_accesses += energy_comps.on_chip_ofms_dynamic_energy
            bram_energy_weights_accesses += energy_comps.on_chip_weights_dynamic_energy

            off_chip_accesses_energy += energy_comps.off_chip_accesses_energy
            dsps_idle_energy += energy_comps.dsps_energy_idle_per_second * exec_time
            dsps_comp_energy += energy_comps.dsps_energy_dynamic

        total_energy = bram_energy_ifms_idle + bram_energy_ofms_idle + bram_energy_weights_idle + \
            bram_energy_ifms_accesses + bram_energy_ofms_accesses + bram_energy_weights_accesses + \
            off_chip_accesses_energy + dsps_idle_energy + dsps_comp_energy

        if print_desc:
            print('bram_energy_ifms_idle: {}\n'.format(bram_energy_ifms_idle),
                'bram_energy_ofms_idle: {}\n'.format(bram_energy_ofms_idle),
                'bram_energy_weights_idle: {}\n'.format(
                    bram_energy_weights_idle),
                'bram_energy_ifms_accesses: {}\n'.format(
                    bram_energy_ifms_accesses),
                'bram_energy_ofms_accesses: {}\n'.format(
                    bram_energy_ofms_accesses),
                'bram_energy_weights_accesses: {}\n'.format(
                    bram_energy_weights_accesses),
                'off_chip_accesses_energy: {}\n'.format(
                    off_chip_accesses_energy),
                'dsps_idle_energy: {}\n'.format(dsps_idle_energy),
                'dsps_comp_energy: {}\n'.format(dsps_comp_energy))

        return total_energy

    @log_function_call
    def get_sub_mappings_pes(self):
        sub_mappings_pes_list = []
        for sub_mapping in self.mapping_list:
            sub_mappings_pes_list.append(sub_mapping.hw_config.num_pes)

        return sub_mappings_pes_list

    def has_heterogeneous_blocks(self):
        first_mapping_type = type(self.mapping_list[0])
        first_mapping_num_ces = len(self.mapping_list[0].get_engines())
        for mapping in self.mapping_list:
            if mapping.has_heterogeneous_blocks() or type(mapping) != first_mapping_type or \
                (isinstance(mapping, SEMLMapping_LBL) and \
                 len(mapping.get_engines()) != first_mapping_num_ces and first_mapping_num_ces == 1):
                return True
            
        return False
    
    def get_block_labels_lis(self):
        blocks = []
        for mapping in self.mapping_list:
            blocks.extend(mapping.get_block_labels_lis())
         
        return list(set(blocks))
    
    def get_minmum_df_possibilities(self):
        min_dfs = []
        for mapping in self.mapping_list:
            min_dfs.extend(mapping.get_minmum_df_possibilities())

        return list(set(min_dfs))
    
    def has_heterogeneous_parallelism_strategies(self):
        first_engine_parallelism_strategy = \
            self.mapping_list[0].get_engines()[0].parallelization_strategy
        for mapping in self.mapping_list:
            if mapping.has_heterogeneous_parallelism_strategies():
                return True
            
            engines = mapping.get_engines()
            for engine in engines:
                if engine.parallelization_strategy != first_engine_parallelism_strategy:
                    return True
            
        return False
    
    def get_parallelism_strategies(self):
        par_strategies = []
        for mapping in self.mapping_list:
            engines = mapping.get_engines()
            for engine in engines:
                par_strategies.append(engine.parallelization_strategy.name)
            
        return list(set(par_strategies))
    
    