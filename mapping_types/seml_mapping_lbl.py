
import __init__
from generic_mapping import GenericMapping
import utils
from mapping_strategies.engines.engine import *
from simple_mapping import SimpleMapping
from basic_mapping import *
from mapping_utils import helper_heuristics
from preformance_record import Metrics
from mapping_utils.logging import *

class SEMLMapping_LBL(SimpleMapping, BasicMapping):

    MAPPING_LABEL = 'SEML_LBL'
    @log_function_call
    def __init__(self, hw_config, model_dag, layers, engines=None,
                 pre_balanced_engines=False,
                 pw_conv_parallelization_strategy=ParallelizationStrategies.OFMS_H_W,
                 first_layer_ifms_are_on_chip=False,
                 last_layer_ofms_are_on_chip=False,
                 dynamic_fm_placement_policy=False,
                 exec_v2=True,
                 adjust_pes = False,
                 enable_multi_ce = False,
                 timing_metric=Metrics.THROUGHPUT):
        super().__init__(hw_config, model_dag, layers, [],
                         first_layer_ifms_are_on_chip, last_layer_ofms_are_on_chip)
        has_dw_layers = utils.has_dw_layers(model_dag, layers[0], len(layers))
        self.pw_conv_parallelization_strategy = pw_conv_parallelization_strategy
        if not pre_balanced_engines:
            engines = []
            if has_dw_layers:
                if len(layers) > 1:
                    if self.pw_conv_parallelization_strategy == ParallelizationStrategies.CUSTOM:
                        engines = [Engine(1, parallelization_strategy=pw_conv_parallelization_strategy), Engine(
                        1, parallelization_strategy=ParallelizationStrategies.CUSTOM_DW)] 
                    else:
                        engines = [Engine(1, parallelization_strategy=pw_conv_parallelization_strategy), Engine(
                        1, parallelization_strategy=ParallelizationStrategies.IN_FILTER_H_W)]  
                    self.dw_engine_index = 1
                else:
                    if self.pw_conv_parallelization_strategy == ParallelizationStrategies.CUSTOM:
                        engines = [Engine(1, parallelization_strategy=ParallelizationStrategies.CUSTOM_DW)] 
                    else:
                        engines = engines = [Engine(1, parallelization_strategy=ParallelizationStrategies.IN_FILTER_H_W)] 
                    self.dw_engine_index = 0
            else:
                engines = [Engine(
                    hw_config.num_pes, parallelization_strategy=pw_conv_parallelization_strategy)]
        else:
            self.dw_engine_index = len(self.engines) - 1

        self.adjust_pes = adjust_pes
        self.engines = engines
        self.mapping_pes = hw_config.num_pes
        self.dynamic_fm_placement_policy = dynamic_fm_placement_policy
        self.num_engines = len(engines)
        self.num_layers = len(layers)
        self.layers_exec_times = [-1] * self.num_layers
        self.multi_ce_layers_exec_times = [-1] * self.num_layers
        self.layer_comp_times = [-1] * self.num_layers
        self.layers_off_chip_fms_access = [-1] * self.num_layers
        self.layers_off_chip_weight_access = [-1] * self.num_layers
        self.layers_on_chip_ifms_access = [-1] * self.num_layers
        self.layers_on_chip_ofms_access = [-1] * self.num_layers
        self.layers_on_chip_weight_access = [-1] * self.num_layers
        self.layer_fms_off_chip = [-1] * self.num_layers
        self.layer_tmp_fms_off_chip = [-1] * self.num_layers
        self.exec_time = -1
        self.tmp_channels_buffer_sz = -1
        self.tmp_channels_data_sz = -1
        self.max_on_chip_fms_layers = -1
        self.exec_v2 = exec_v2
        self.enable_multi_ce = enable_multi_ce
        self.timing_metric = timing_metric
        if self.enable_multi_ce:
            self.multi_ce_engines = []
            self.multi_ce_layers = []
        else:
            self.multi_ce_engines = None
            self.multi_ce_layers = None
        self.allocate_and_balance_pes_to_engines(hw_config.num_pes)

    @log_function_call
    def get_label(self):
        return self.MAPPING_LABEL

    @log_function_call
    def get_num_engines(self):
        if self.num_engines > 1 and utils.has_dw_layers(self.model_dag, self.layers[0], len(self.layers)):
            return self.num_engines / 2

        return self.num_engines
    
    @log_function_call
    def allocate_and_balance_pes_to_engines_unpipelined(self, num_pes):
        op_counts = utils.get_layers_op_counts_by_indices(
            self.model_dag, self.layers)
        dw_ops = 0
        conv_ops = 0

        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            if utils.is_dw_layer(layer_specs):
                dw_ops += op_counts[i]
            else:
                conv_ops += op_counts[i]

        self.engines[0].num_pes = math.ceil(
            num_pes * conv_ops / (conv_ops + dw_ops))
        self.engines[0].distribute_PEs_on_dims(
            self.model_dag, self.layers, targeted_layer_type=['pw', 's'])

        if len(self.engines) > 1:
            self.engines[1].num_pes = num_pes - self.engines[0].num_pes
            self.engines[1].distribute_PEs_on_dims(
                self.model_dag, self.layers, targeted_layer_type=['dw'])

    @log_function_call
    def split_layer_fms_on_a_dim(self, layer_specs, split_dim_index, split_ratios_list):
        split_dicts = []
        split_keys_list = ['ifms_shape', 'ofms_shape']
        
        for split_key_index, split_key in enumerate(split_keys_list):
            splits_so_far = 0
            to_split_dim_val = 0
            for i, split_ratio in enumerate(split_ratios_list):
                if split_key_index == 0:
                    split_dicts.append(utils.copy_dict(layer_specs))
                if split_dicts[i] == None:
                    continue
                for inner_dim_index in range(3):
                    dims = layer_specs[split_key]
                    if inner_dim_index == 0:
                        split_dicts[i][split_key] = []
                    dim_val = dims[inner_dim_index]
                    if inner_dim_index != split_dim_index or split_key == 'ifms_shape' and split_dim_index == 0:
                        split_dicts[i][split_key].append(dim_val)
                    else:
                        to_split_dim_val = dim_val
                        portion = min(math.ceil(dim_val * split_ratio), dim_val - splits_so_far)
                        splits_so_far += portion
                        split_dicts[i][split_key].append(portion)

            assert splits_so_far == to_split_dim_val
        
        for i, split_dict in enumerate(split_dicts):
            empty_dim = False
            for key, val in split_dict.items():
                if key in split_keys_list and 0 in val:
                    empty_dim = True
                    break
            if empty_dim:
                split_dicts[i] = None

        return split_dicts

    @log_function_call
    def multi_ce_decide_best_split(self, unsplited_time):

        all_pes = self.engines[0].num_pes
        distributed_pes_list = helper_heuristics.decompose_into_sum_of_powers_of_two(all_pes)
        sum_distributed_pes = sum(distributed_pes_list)
        split_ratios_list = [distributed_pes / sum_distributed_pes for distributed_pes in distributed_pes_list]
        for engine_pes in distributed_pes_list:
            self.multi_ce_engines.append(Engine(engine_pes, parallelization_strategy= self.pw_conv_parallelization_strategy))
        layer_specs_list_best = []
        best_exec_time = math.inf
        layer_exec_times = []
        for dim_index in range(3):
            layer_specs_list = []
            for engine_index in range(len(distributed_pes_list)):
                layer_specs_list.append([])
            for layer_specs in self.model_dag:
                if utils.is_conv_layer(layer_specs):
                    split_dicts = self.split_layer_fms_on_a_dim(layer_specs, dim_index, split_ratios_list)
                    for engine_index in range(len(distributed_pes_list)):
                        layer_specs_list[engine_index].append(split_dicts[engine_index])
                else:
                    for engine_index in range(len(distributed_pes_list)):
                        layer_specs_list[engine_index].append(None)
            
            for engine_index, engine in enumerate(self.multi_ce_engines):
                layer_exec_times.append([])
                if engine_index != 0:
                    engine.parallelization_strategy = self.multi_ce_engines[0].parallelization_strategy

                engine.distribute_PEs_on_dims(
                    layer_specs_list[engine_index], self.layers, targeted_layer_type=['pw', 's'])
                for layer_index in self.layers:
                    layer_specs = layer_specs_list[engine_index][layer_index]
                    #print(engine_index, layer_specs)
                    if layer_specs is not None and not utils.is_dw_layer(layer_specs):
                        layer_exec_times[-1].append(engine.calc_layer_exec_time(layer_specs))
                    else:
                        layer_exec_times[-1].append(0)

            current_exec_time = 0
            #print('###', layer_exec_times[0])
            for i in range(len(self.layers)):
                layer_exec_time = 0
                for j in range(len(self.multi_ce_engines)):
                    layer_exec_time = max(layer_exec_time, layer_exec_times[j][i])
                
                current_exec_time += layer_exec_time
                self.multi_ce_layers_exec_times[i] = layer_exec_time
            
            if current_exec_time < best_exec_time:
                best_exec_time = current_exec_time
                layer_specs_list_best = []
                for engine_index in range(len(distributed_pes_list)):
                    layer_specs_list_best.append([])
                    for layer_index in range(len(layer_specs_list[0])):
                        layer_specs_list_best[-1].append(layer_specs_list[engine_index][layer_index])

        #print('>>>>>>',unsplited_time / best_exec_time)
        if best_exec_time >= unsplited_time:
            self.multi_ce_engines = None
            self.multi_ce_layers = None
            self.enable_multi_ce = False
        else:
            self.multi_ce_layers = []
            for engine_index, engine in enumerate(self.multi_ce_engines):
                self.multi_ce_layers.append([])
                if engine_index != 0:
                    engine.parallelization_strategy = self.multi_ce_engines[0].parallelization_strategy
                engine.distribute_PEs_on_dims(
                    layer_specs_list[engine_index], self.layers, targeted_layer_type=['pw', 's'])
                for layer_index in range(len(layer_specs_list_best[0])):    
                    self.multi_ce_layers[-1].append(layer_specs_list_best[engine_index][layer_index])

    @log_function_call
    def allocate_and_balance_pes_to_engines(self, num_pes):
        op_counts = utils.get_layers_op_counts_by_indices(
            self.model_dag, self.layers)
        max_dw_ops = 0
        max_conv_ops = 0
        min_to_keep_pes_for_dw = 0
        if not self.adjust_pes or not utils.has_dw_layers(self.model_dag):
            for i in range(len(self.layers)):
                layer_index = self.layers[i]
                layer_specs = self.model_dag[layer_index]
                if utils.is_dw_layer(layer_specs):
                    max_dw_ops = max(max_dw_ops, op_counts[i])
                    min_to_keep_pes_for_dw = 1
                else:
                    max_conv_ops = max(max_conv_ops, op_counts[i])

            if max_conv_ops > 0:
                self.engines[0].num_pes = max(min(math.ceil(
                    num_pes * max_conv_ops / (max_conv_ops + max_dw_ops)), num_pes - min_to_keep_pes_for_dw), 1)
                self.engines[0].distribute_PEs_on_dims(
                    self.model_dag, self.layers, targeted_layer_type=['pw', 's'])
            else:
                self.engines[0].num_pes = num_pes
                self.engines[0].distribute_PEs_on_dims(
                    self.model_dag, self.layers, targeted_layer_type=['dw'])

            if len(self.engines) > 1:
                self.engines[1].num_pes = num_pes - self.engines[0].num_pes
                self.engines[1].distribute_PEs_on_dims(
                    self.model_dag, self.layers, targeted_layer_type=['dw'])
                
        else:
            pw_s_ops = utils.get_layers_op_counts_by_indices(self.model_dag, self.layers, ['s', 'pw'])
            dw_ops = utils.get_layers_op_counts_by_indices(self.model_dag, self.layers, ['dw'])
            max_len = max(len(pw_s_ops), len(dw_ops))
            for i in range(len(pw_s_ops), max_len):
                pw_s_ops.append(0)
            for i in range(len(dw_ops), max_len):
                dw_ops.append(0)
            [pw_s_pes, dw_pes], _ = helper_heuristics.minimize_max_ratio_exh([pw_s_ops, dw_ops], self.hw_config.num_pes)
            
            if sum(pw_s_ops) > 0:
                self.engines[0].num_pes = max(pw_s_pes, 1)
                self.engines[0].distribute_PEs_on_dims(
                    self.model_dag, self.layers, targeted_layer_type=['pw', 's'])
            else:
                self.engines[0].num_pes = num_pes
                self.engines[0].distribute_PEs_on_dims(
                    self.model_dag, self.layers, targeted_layer_type=['dw'])

        if utils.is_pow_of_2(self.engines[0].num_pes):
            self.multi_ce_engines = None
            self.multi_ce_layers = None
            self.enable_multi_ce = False
        elif self.enable_multi_ce:
            initial_exec_time = 0
            for i in range(len(self.layers)):
                layer_index = self.layers[i]
                layer_specs = self.model_dag[layer_index]
                if not utils.is_dw_layer(layer_specs):
                    initial_exec_time += \
                        self.engines[0].calc_layer_exec_time(
                            layer_specs)
            
            self.multi_ce_decide_best_split(initial_exec_time)

        if self.adjust_pes and utils.has_dw_layers(self.model_dag):
            if len(self.engines) > 1:
                initial_exec_time = math.inf
                while dw_pes >= 1:
                    self.engines[1].num_pes = max(dw_pes, 1)
                    self.engines[1].distribute_PEs_on_dims(
                        self.model_dag, self.layers, targeted_layer_type=['dw'])
                    current_exec_time = self.calc_exec_time()
                    if current_exec_time <= initial_exec_time:
                        initial_exec_time = current_exec_time
                        dw_pes //= 2
                    else:
                        dw_pes *= 2
                        self.engines[1].num_pes = max(dw_pes, 1)
                        self.engines[1].distribute_PEs_on_dims(
                            self.model_dag, self.layers, targeted_layer_type=['dw'])
                        break
                    
    @log_function_call
    def calc_layer_compute_times(self):
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            if utils.is_dw_layer(layer_specs):
                self.layer_comp_times[i] = \
                    self.engines[self.dw_engine_index].calc_layer_exec_time(
                        layer_specs) / self.hw_config.frequency
            else:
                if self.enable_multi_ce:
                    self.layer_comp_times[i] = self.multi_ce_layers_exec_times[i] / self.hw_config.frequency
                else:
                    self.layer_comp_times[i] = \
                        self.engines[0].calc_layer_exec_time(
                            layer_specs) / self.hw_config.frequency

        return self.layer_comp_times

    @log_function_call
    def calc_layer_off_chip_access_time(self, layer_index):
        if self.layers_off_chip_fms_access[layer_index] == -1 or self.layers_off_chip_weight_access[layer_index] == -1:
            self.calc_off_chip_fms_and_weights_access_intra()

        return (self.layers_off_chip_fms_access[layer_index] + self.layers_off_chip_weight_access[layer_index]) / self.hw_config.bw

    @log_function_call
    def calc_off_chip_access_time_only(self):
        access_time = 0
        for i in range(len(self.layers)):
            access_time += self.calc_layer_off_chip_access_time(i)

        return access_time

    @log_function_call
    def calc_compute_time_only(self):
        return sum(self.calc_layer_compute_times())

    @log_function_call
    def calc_exec_time(self, print_desc=False):

        if self.exec_v2:
            return self.calc_exec_time_v2(print_desc)

        if print_desc:
            print(self.MAPPING_LABEL)

        self.calc_layer_compute_times()
        exec_time = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            self.layers_exec_times[i] = max(
                self.layer_comp_times[i], self.calc_layer_off_chip_access_time(i))
            # assuming engines are pipelined
            if utils.is_dw_layer(layer_specs):
                if i + 1 < len(self.layers):
                    exec_time += max(self.layers_exec_times[i],
                                     self.layers_exec_times[i + 1])
                else:
                    exec_time += self.layers_exec_times[i]
            elif i > 0 and not utils.is_dw_layer(self.model_dag[self.layers[i - 1]]):
                exec_time += self.layers_exec_times[i]

        self.exec_time = exec_time
        return exec_time

    @log_function_call
    def calc_exec_time_v2(self, print_desc=False):

        if print_desc:
            print(self.MAPPING_LABEL, 'V2')

        self.calc_layer_compute_times()
        exec_time = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            self.layers_exec_times[i] = max(
                self.layer_comp_times[i], self.calc_layer_off_chip_access_time(i))
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            # assuming engines are pipelined
            if utils.is_dw_layer(layer_specs):
                if i + 1 < len(self.layers):
                    exec_time += max(self.layers_exec_times[i],
                                     self.layers_exec_times[i + 1])
                else:
                    exec_time += self.layers_exec_times[i]
            elif i == 0 or (i > 0 and not utils.is_dw_layer(self.model_dag[self.layers[i - 1]]) or not utils.has_dw_layers(self.model_dag)):
                exec_time += self.layers_exec_times[i]

        self.exec_time = exec_time
        return exec_time

    @log_function_call
    def get_segment_exec_times():
        pass

    @log_function_call
    def calc_throughput(self):
        if self.exec_time == -1:
            self.exec_time = self.calc_exec_time()

        return 1 / self.exec_time

    @log_function_call
    def calc_fms_buffer_sz_intra(self):
        fms_buffer_sz = 2 * constants.KiB
        fms_buffer_data_size = fms_buffer_sz
        weights_buffer_sz = self.calc_weights_buffer_sz()
        for i in range(len(self.layers) - 1):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            ifm_ofm_size = max(utils.get_layer_ifms_size(
                layer_specs), utils.get_layer_ofms_size(layer_specs))
            ifm_ofm_data_size = ifm_ofm_size
            ifm_ofm_size = self.calc_actual_bram_cons(
                ifm_ofm_size, self.engines[0].get_parallelism_fms())
            ifm_ofm_size *= 2
            if ifm_ofm_size < self.hw_config.on_chip_memory:
                if ifm_ofm_size > fms_buffer_sz:
                    self.max_on_chip_fms_layers = layer_index
                    fms_buffer_data_size = ifm_ofm_data_size
                    fms_buffer_sz = ifm_ofm_size
                self.layer_fms_off_chip[i] = 0
            else:
                self.layer_fms_off_chip[i] = 1

        if fms_buffer_sz == 0:
            fms_buffer_sz = max(2 * constants.KiB, self.hw_config.on_chip_memory - weights_buffer_sz)
        self.layer_fms_off_chip[-1] = 0
        self.fms_buffer_data_size = fms_buffer_data_size

        return fms_buffer_sz

    @log_function_call
    def calc_tmp_fms_buffer_sz(self):
        fms_buffer_sz = self.calc_main_fms_buffer_sz()
        weights_buffer_sz = self.calc_weights_buffer_sz()
        tmp_channels_buffer_sz = 0
        for i, layer in enumerate(self.layers):
            tmp_channels_sz = 0
            layer_specs = self.model_dag[layer]
            layer_chidren = utils.get_layer_children_with_fusion(
                self.model_dag, layer_specs)
            if len(layer_chidren) > 1:
                tmp_channels_sz = utils.get_layer_ofms_size(layer_specs)

            tmp_channels_buffer_sz = self.calc_actual_bram_cons(
                tmp_channels_sz, self.engines[0].get_parallelism_fms())
            if self.hw_config.on_chip_memory >= fms_buffer_sz + weights_buffer_sz + tmp_channels_buffer_sz:
                self.tmp_channels_buffer_sz = max(
                    self.tmp_channels_buffer_sz, tmp_channels_buffer_sz)

                self.tmp_channels_data_sz = max(
                    self.tmp_channels_data_sz, tmp_channels_sz)
                self.layer_tmp_fms_off_chip[i] = 0
            else:
                self.layer_tmp_fms_off_chip[i] = min(1, tmp_channels_sz)

        return self.tmp_channels_buffer_sz

    @log_function_call
    def calc_main_fms_buffer_sz(self):
        on_chip_memory = self.hw_config.on_chip_memory
        intra_fms_buffer_sz = self.calc_fms_buffer_sz_intra()
        weights_buffer_sz = self.calc_weights_buffer_sz()

        first_layer_ifms_size = utils.get_layer_ifms_size(
            self.model_dag[self.layers[0]])
        last_layer_ofms_size = utils.get_layer_ofms_size(
            self.model_dag[self.layers[-1]])
        # The priority is for weights and intermediate results, as they are more on the critical path
        # if there is a space left, then ...
        avialable_on_chip_memoy = on_chip_memory - \
            (intra_fms_buffer_sz + weights_buffer_sz)
        iofms_on_chip_sz = \
            self.on_chip_buffer_sz_for_first_last_iofms(avialable_on_chip_memoy,
                                                        first_layer_ifms_size, last_layer_ofms_size)

        return intra_fms_buffer_sz + iofms_on_chip_sz

    @log_function_call
    def calc_fms_buffer_sz(self, print_desc=False):
        if print_desc:
            print(self.MAPPING_LABEL, 'calc_fms_buffer_sz',
                  self.calc_main_fms_buffer_sz(), self.calc_tmp_fms_buffer_sz())
        return self.calc_main_fms_buffer_sz() + self.calc_tmp_fms_buffer_sz()

    @log_function_call
    def calc_weights_buffer_sz(self):
        weights_buffer_sz = 0
        dw_weights_buffer_sz = 0
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            filter_dim = utils.get_filter_dim(layer_specs)
            if utils.is_dw_layer(layer_specs):
                dw_weights_buffer_sz = max(dw_weights_buffer_sz,
                                           utils.get_layer_ifms_shape(layer_specs)[0] * filter_dim * filter_dim)
            else:
                weights_buffer_sz = max(weights_buffer_sz,
                                        utils.get_layer_ifms_shape(layer_specs)[0] * filter_dim * filter_dim *
                                        self.engines[0].par_ofms)
                # print(weights_buffer_sz, utils.get_layer_ifms_shape(layer_specs)[0],
                #       self.engines[self.dw_engine_index].par_ofms, layer_index)

        weights_buffer_sz = self.calc_actual_bram_cons(
            weights_buffer_sz, self.engines[0].get_ports_weights())
        if len(self.engines) > 1:
            dw_weights_buffer_sz = self.calc_actual_bram_cons(
                dw_weights_buffer_sz, 1)

        # double buffering
        return dw_weights_buffer_sz + 2 * weights_buffer_sz

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space
    @log_function_call
    def calc_off_chip_weights_access(self):
        all_layer_weight_accesses = sum(self.layers_off_chip_weight_access)
        if all_layer_weight_accesses < 0:
            self.calc_off_chip_fms_and_weights_access_intra()
            all_layer_weight_accesses = sum(self.layers_off_chip_weight_access)
        
        return all_layer_weight_accesses

    # the assumption is that the priority for storing the intermediate fms, then the weights if there is space,
    # then the ifms of the first layer and the ofms of the last layer
    # however, if weights are stored off-chip due to not fitting, then the ifms of the first layer and the ofms
    # of the last layer could be stored on-chip
    @log_function_call
    def calc_off_chip_fms_and_weights_access_intra(self):
        weights_buffer_sz = self.calc_weights_buffer_sz()
        self.calc_fms_buffer_sz_intra()
        ifms_data_sz_intra = self.fms_buffer_data_size / 2
        if sum(self.layer_fms_off_chip) < 0:
            self.calc_fms_buffer_sz_intra()
        if sum(self.layer_tmp_fms_off_chip) < 0:
            self.calc_tmp_fms_buffer_sz()
        fms_off_chip = sum(self.layer_fms_off_chip) > 0 and not self.dynamic_fm_placement_policy
        tmp_fms_off_chip = sum(self.layer_tmp_fms_off_chip) > 0 and not self.dynamic_fm_placement_policy
        for i in range(len(self.layers)):
            layer_index = self.layers[i]
            layer_specs = self.model_dag[layer_index]
            layer_ifms_sz_act = utils.get_layer_ifms_size(
                layer_specs)
            layer_ifms_sz = layer_ifms_sz_act if i > 0 else 0
            layer_ofms_sz_act = utils.get_layer_ofms_size(
                layer_specs)
            layer_ofms_sz = layer_ofms_sz_act if i < len(self.layers) - 1 else 0
            layer_weights_size = utils.get_layer_weights_size(layer_specs)
            layer_num_filters = utils.get_layer_weights_shape(layer_specs)[0]
            num_weight_tiles_per_ifm = math.ceil(layer_num_filters / self.engines[0].par_ofms)
            if utils.is_dw_layer(layer_specs):
                num_weight_tiles_per_ifm = 1
            num_fm_tiles = math.ceil(layer_ifms_sz_act / ifms_data_sz_intra)
            layer_chidren = utils.get_layer_children_with_fusion(
                self.model_dag, layer_specs)
            if self.layer_fms_off_chip[i] == 1 or fms_off_chip or \
                ((self.layer_tmp_fms_off_chip[i] == 1 or tmp_fms_off_chip) and len(layer_chidren) > 1):
                os_l_ifms_stationary_weight_accesses = layer_weights_size * num_fm_tiles
                os_l_weights_stationary_ifms_accesses = layer_ifms_sz * num_weight_tiles_per_ifm
                if layer_ifms_sz + os_l_ifms_stationary_weight_accesses <= \
                        layer_weights_size + os_l_weights_stationary_ifms_accesses:
                    self.layers_off_chip_fms_access[i] = layer_ofms_sz + \
                        layer_ifms_sz
                    self.layers_off_chip_weight_access[i] = os_l_ifms_stationary_weight_accesses
                else:
                    self.layers_off_chip_fms_access[i] = layer_ofms_sz + \
                        os_l_weights_stationary_ifms_accesses
                    self.layers_off_chip_weight_access[i] = layer_weights_size
            else:
                self.layers_off_chip_fms_access[i] = 0
                self.layers_off_chip_weight_access[i] = layer_weights_size
            
            self.layers_on_chip_ofms_access[i] = layer_ofms_sz
            self.layers_on_chip_ifms_access[i] = layer_ifms_sz * num_weight_tiles_per_ifm
            self.layers_on_chip_weight_access[i] = layer_weights_size * num_fm_tiles

    @log_function_call
    def calc_off_chip_fms_access(self, print_desc=False):
        off_chip_accesses = sum(self.layers_off_chip_fms_access)
        fms_buffer_sz = self.calc_fms_buffer_sz_intra()

        if off_chip_accesses < 0:
            self.calc_off_chip_fms_and_weights_access_intra()
            off_chip_accesses = sum(self.layers_off_chip_fms_access)

        weights_buffer_sz = self.calc_weights_buffer_sz()
        available_on_chip = self.hw_config.on_chip_memory - \
            (weights_buffer_sz + fms_buffer_sz)

        layer_specs = self.model_dag[self.layers[0]]
        first_layer_ifms_sz = utils.get_layer_ifms_size(layer_specs)
        layer_specs = self.model_dag[self.layers[-1]]
        last_layer_ifms_sz = utils.get_layer_ofms_size(layer_specs)

        if self.off_chip_fms_access_of_first_and_last_layers < 0:
            self.on_chip_buffer_sz_for_first_last_iofms(available_on_chip,
                                                        first_layer_ifms_sz,
                                                        last_layer_ifms_sz)

        off_chip_accesses += self.off_chip_fms_access_of_first_and_last_layers

        return off_chip_accesses

    @log_function_call
    def get_off_chip_tmp_channels_layers(self):
        are_tmp_channels_on_chip = []
        if self.tmp_channels_data_sz < 0:
            self.calc_tmp_fms_buffer_sz()
        for layer in self.layers:
            layer_specs = self.model_dag[layer]
            layer_chidren = utils.get_layer_children_with_fusion(
                self.model_dag, layer_specs)
            if len(layer_chidren) > 1:
                if self.calc_actual_bram_cons(
                        utils.get_layer_ofms_size(layer_specs), self.engines[0].get_parallelism_fms()) > self.tmp_channels_buffer_sz:
                    are_tmp_channels_on_chip.append(layer)

        return are_tmp_channels_on_chip

    @log_function_call
    def get_dict_representation(self):
        mapping_dict = {}
        num_layers = len(self.num_layers)
        layers_str = '0'
        if num_layers > 1:
            layers_str += '-{}'.format(num_layers)
        engines_str = '0'

        mapping_dict[layers_str] = engines_str

        return mapping_dict

    @log_function_call
    def calc_on_chip_accesses(self):
        if self.layers_on_chip_ifms_access[0] == -1:
            self.calc_off_chip_fms_and_weights_access_intra()

        return sum(self.layers_on_chip_weight_access), sum(self.layers_on_chip_ifms_access), sum(self.layers_on_chip_ofms_access)
    
    def has_heterogeneous_blocks(self):
        return self.multi_ce_engines is not None
    
    def get_block_labels_lis(self):
        blocks = ['S']
        if self.multi_ce_engines is not None:
            blocks = ['Pa']
        if self.num_engines > 1:
            blocks.append('Pi')
        
        return blocks

    def get_minmum_df_possibilities(self):
        self.calc_off_chip_fms_and_weights_access_intra()
        if self.num_engines > 1: #pw and dw
            min_dfs = ['OS', 'WS']  
        else:
            if sum(self.layers_off_chip_fms_access) == 0:
                min_dfs = ['WS_LOS']
            else:
                min_dfs = ['OS']
        
        if self.multi_ce_engines is not None: #parallel CEs 
            min_dfs.append('OS_LWS')

        return list(set(min_dfs))
    
    def has_heterogeneous_parallelism_strategies(self):
        engines = self.get_engines()
        first_paralleism_strategy = engines[0].parallelization_strategy
        for engine in engines:
            if engine.parallelization_strategy != first_paralleism_strategy:
                return True
            
        if self.multi_ce_engines is not None:
            for engine in self.multi_ce_engines:
                if engine.parallelization_strategy != first_paralleism_strategy:
                    return True
            
        return False
    
    def get_parallelism_strategies(self):
        par_strategies = []
        engines = self.get_engines()
        for engine in engines:
            par_strategies.append(engine.parallelization_strategy.name)
            
        if self.multi_ce_engines is not None:
            for engine in self.multi_ce_engines:
                par_strategies.append(engine.parallelization_strategy.name)
                        
        return list(set(par_strategies))