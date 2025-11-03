import __init__
import mapping_utils.mapping_general_utils as mapping_general_utils
from mapping_types.mapping_description import *
from mapping_utils import custom_mapping_utils
import constants
import random
from mapping_utils import mapping_exec_utils
import mapping_utils.helper_heuristics as helper_heuristics
import time
import os
from hw_config import *


def empty_clusters(cluster_layers):
    for num_layers in cluster_layers:
        if num_layers == 0:
            return True

    return False


def adjust_with_the_unused_pes(mapping, model_dag, ordered_engines_pes, hw_config,
                               cluster_layer_count_map,
                               board_name, model_name,
                               opt_metric, timing_metric):

    op_counts = utils.get_layers_op_counts(model_dag, 0)
    segment_exec_times = mapping.get_segment_exec_times()
    unused_pes = hw_config.num_pes - sum(ordered_engines_pes)
    clusters_layer_indices = [0]
    new_engines_info = {}
    for layer_count in cluster_layer_count_map:
        clusters_layer_indices.append(
            clusters_layer_indices[-1] + layer_count)

    while unused_pes > 0:
        max_exec_time = max(segment_exec_times)
        max_exec_time_index = segment_exec_times.index(
            max_exec_time)
        if cluster_layer_count_map[max_exec_time_index] == 1:
            break
        next_exec_time = 0
        prev_exec_time = 0
        if max_exec_time_index < len(ordered_engines_pes) - 1:
            next_exec_time = segment_exec_times[max_exec_time_index + 1]
        if max_exec_time_index > 0:
            prev_exec_time = segment_exec_times[max_exec_time_index - 1]

        added = False
        if next_exec_time > prev_exec_time:
            candidate_layer_index = clusters_layer_indices[max_exec_time_index + 1]
            candidate_layer_ops = op_counts[candidate_layer_index]
            required_pes = max(1, utils.pow_of_2_geq(
                candidate_layer_ops / (max_exec_time * hw_config.frequency)))
            if required_pes < unused_pes:
                added = True
                unused_pes -= required_pes
                if max_exec_time_index + 1 not in new_engines_info:
                    new_engines_info[max_exec_time_index + 1] = {}
                    new_engines_info[max_exec_time_index +
                                     1]['layers'] = [candidate_layer_index]
                    new_engines_info[max_exec_time_index +
                                     1]['pes'] = required_pes
                else:
                    new_engines_info[max_exec_time_index +
                                     1]['layers'].append(candidate_layer_index)
                    new_engines_info[max_exec_time_index +
                                     1]['pes'] += required_pes
        elif prev_exec_time != 0 and not added:
            # last layer of prev CE
            candidate_layer_index = clusters_layer_indices[max_exec_time_index] - 1
            candidate_layer_ops = op_counts[candidate_layer_index]
            required_pes = max(1, utils.pow_of_2_geq(
                candidate_layer_ops / (max_exec_time * hw_config.frequency)))
            if required_pes < unused_pes:
                added = True
                unused_pes -= required_pes
                if max_exec_time_index - 1 not in new_engines_info:
                    new_engines_info[max_exec_time_index - 1] = {}
                    new_engines_info[max_exec_time_index -
                                     1]['layers'] = [candidate_layer_index]
                    new_engines_info[max_exec_time_index -
                                     1]['pes'] = required_pes
                else:
                    new_engines_info[max_exec_time_index -
                                     1]['layers'].append(candidate_layer_index)
                    new_engines_info[max_exec_time_index -
                                     1]['pes'] += required_pes

        if added:
            segment_exec_times[max_exec_time_index] -= candidate_layer_ops / \
                (ordered_engines_pes[max_exec_time_index]
                    * hw_config.frequency)
            cluster_layer_count_map[max_exec_time_index] -= 1
        else:
            break

    new_ordered_pes = []
    new_cluster_layer_counts = []
    for i in range(len(ordered_engines_pes)):
        original_first_layer = clusters_layer_indices[i]
        first_new_layer = -1
        if i in new_engines_info:
            info = new_engines_info[i]
            first_new_layer = info['layers'][0]
            new_pes = info['pes']
            new_layers = len(info['layers'])

        if first_new_layer != -1 and first_new_layer < original_first_layer:
            new_ordered_pes.append(new_pes)
            new_cluster_layer_counts.append(new_layers)
        else:
            new_ordered_pes.append(ordered_engines_pes[i])
            new_cluster_layer_counts.append(
                cluster_layer_count_map[i])
            if first_new_layer != -1:
                new_ordered_pes.append(new_pes)
                new_cluster_layer_counts.append(new_layers)

    num_layers_so_far = 0
    num_engines_so_far = 0
    new_mapping_config_dict = {}
    for cluster_index in range(len(new_ordered_pes)):
        current_cluster_num_layers = new_cluster_layer_counts[cluster_index]
        last_layer_offset = num_layers_so_far + current_cluster_num_layers
        if current_cluster_num_layers == 1:
            new_mapping_config_dict["{}".format(num_layers_so_far,
                                                last_layer_offset)] = "{}".format(num_engines_so_far)
        else:
            new_mapping_config_dict["{}-{}".format(num_layers_so_far,
                                                   last_layer_offset)] = "{}".format(num_engines_so_far)

        num_layers_so_far += current_cluster_num_layers
        num_engines_so_far += 1

    mapping = custom_mapping_utils.custom_mapping_from_desc_dict(board_name, model_dag,
                                                                 new_mapping_config_dict, timing_metric, adjust_pes=False,
                                                                 adjusted_pes_list=new_ordered_pes)

    perf_record = mapping_exec_utils.run_mapping(
        board_name, model_name, mapping)

    perf_val = perf_record.get_metric_val(opt_metric)

    return perf_val, new_mapping_config_dict, new_cluster_layer_counts, new_ordered_pes, mapping


def adjust_by_shifting(mapping, model_dag, ordered_engines_pes, hw_config,
                       cluster_layer_count_map,
                       board_name, model_name,
                       opt_metric, timing_metric):
    vitsited = 0
    op_counts = utils.get_layers_op_counts(model_dag, 0)

    while vitsited < len(cluster_layer_count_map):
        clusters_layer_indices = [0]
        for layer_count in cluster_layer_count_map:
            clusters_layer_indices.append(
                clusters_layer_indices[-1] + layer_count)
        segment_exec_times = mapping.get_segment_exec_times()
        max_exec_time = max(segment_exec_times)
        max_exec_time_index = segment_exec_times.index(
            max_exec_time)
        if cluster_layer_count_map[max_exec_time_index] == 1:
            break
        next_exec_time = 0
        prev_exec_time = 0
        to_move_layer_to_next = -1
        to_move_layer_to_prev = -1
        if max_exec_time_index < len(ordered_engines_pes) - 1:
            next_exec_time = segment_exec_times[max_exec_time_index + 1]
            # last layer
            to_move_layer_to_next = clusters_layer_indices[max_exec_time_index + 1] - 1
            to_move_ops = op_counts[to_move_layer_to_next]
            new_next_exec_time = next_exec_time + to_move_ops / \
                (ordered_engines_pes[max_exec_time_index + 1]
                    * hw_config.frequency)
            if new_next_exec_time > max_exec_time:
                to_move_layer_to_next = -1
        if max_exec_time_index > 0:
            prev_exec_time = segment_exec_times[max_exec_time_index - 1]
            # first layer
            to_move_layer_to_prev = clusters_layer_indices[max_exec_time_index]
            to_move_ops = op_counts[to_move_layer_to_prev]
            new_prev_exec_time = prev_exec_time + to_move_ops / \
                (ordered_engines_pes[max_exec_time_index - 1]
                    * hw_config.frequency)
            if new_prev_exec_time > max_exec_time:
                to_move_layer_to_prev = -1

        if to_move_layer_to_next != -1 and to_move_layer_to_prev != -1:
            vitsited += 1
            if new_next_exec_time < new_prev_exec_time:
                cluster_layer_count_map[max_exec_time_index + 1] += 1
                cluster_layer_count_map[max_exec_time_index] -= 1
        elif to_move_layer_to_prev != -1:
            vitsited += 1
            cluster_layer_count_map[max_exec_time_index - 1] += 1
            cluster_layer_count_map[max_exec_time_index] -= 1
        else:
            break

    num_layers_so_far = 0
    num_engines_so_far = 0
    new_mapping_config_dict = {}
    for cluster_index in range(len(ordered_engines_pes)):
        current_cluster_num_layers = cluster_layer_count_map[cluster_index]
        last_layer_offset = num_layers_so_far + current_cluster_num_layers
        if current_cluster_num_layers == 1:
            new_mapping_config_dict["{}".format(num_layers_so_far,
                                                last_layer_offset)] = "{}".format(num_engines_so_far)
        else:
            new_mapping_config_dict["{}-{}".format(num_layers_so_far,
                                                   last_layer_offset)] = "{}".format(num_engines_so_far)

        num_layers_so_far += current_cluster_num_layers
        num_engines_so_far += 1

    mapping = custom_mapping_utils.custom_mapping_from_desc_dict(board_name, model_dag,
                                                                 new_mapping_config_dict, timing_metric, adjust_pes=False,
                                                                 adjusted_pes_list=ordered_engines_pes)

    perf_record = mapping_exec_utils.run_mapping(
        board_name, model_name, mapping)

    perf_val = perf_record.get_metric_val(opt_metric)

    return perf_val, new_mapping_config_dict, cluster_layer_count_map, ordered_engines_pes


def run_hiera_map(mapping_desc, opt_metric, max_clusters):
    if not constants.TEST_SEEDS:
        random.seed(5)

    board_name = mapping_desc.board_name
    model_name = mapping_desc.model_name
    model_dag = mapping_desc.model_dag
    timing_metric = mapping_desc.timing_metric
    hw_config = HWConfig(board_name)

    opt_val = -1
    opt_mapping_dict = None
    opt_val_adjusted = -1
    opt_mapping_dict_adjusted = None
    print('run_hiera_map', board_name, model_name)
    t0 = time.time()

    if opt_metric != Metrics.NONE:
        features_mask_dict = helper_heuristics.features_mask_dict.keys()
        for kmeans_mak_key in features_mask_dict:
            for num_clusters in range(constants.MIN_CLUSTERS, max_clusters + 1):
                cluster_layer_count_map, silhouette_score = helper_heuristics.cluster_layers_using_kmeans(
                    model_dag, num_clusters, features_version=2, feature_mask_key=kmeans_mak_key)
                if empty_clusters(cluster_layer_count_map):
                    break
                possibilities = 2 ** num_clusters
                for possibility_index in range(possibilities):
                    possibility_binary = mapping_general_utils.dec_to_binary_as_list(
                        possibility_index, num_clusters)
                    num_layers_so_far = 0
                    num_engines_so_far = 0
                    mapping_config_dict = {}
                    for cluster_index in range(num_clusters):
                        current_cluster_num_layers = cluster_layer_count_map[cluster_index]
                        current_cluster_num_engines = 1
                        last_layer_offset = num_layers_so_far + current_cluster_num_layers
                        if current_cluster_num_layers == 1:
                            mapping_config_dict["{}".format(num_layers_so_far,
                                                            last_layer_offset)] = "{}".format(num_engines_so_far)
                        elif possibility_binary[cluster_index] == 0:
                            mapping_config_dict["{}-{}".format(num_layers_so_far,
                                                               last_layer_offset)] = "{}".format(num_engines_so_far)
                        else:
                            current_cluster_num_engines = current_cluster_num_layers
                            last_engine_offset = num_engines_so_far + current_cluster_num_layers
                            mapping_config_dict["{}-{}".format(num_layers_so_far,
                                                               last_layer_offset)] = \
                                "{}-{}".format(num_engines_so_far,
                                               last_engine_offset)

                        num_layers_so_far += current_cluster_num_layers
                        num_engines_so_far += current_cluster_num_engines
                    
                    mapping = custom_mapping_utils.custom_mapping_from_desc_dict(board_name, model_dag,
                                                                                 mapping_config_dict, timing_metric,
                                                                                 adjust_pes=True)

                    perf_record = mapping_exec_utils.run_mapping(
                        board_name, model_name, mapping)
                    current_val = perf_record.get_metric_val(opt_metric)
                    if opt_val == -1 or first_better_than_second(opt_metric, current_val, opt_val):
                        if not utils.has_dw_layers(model_dag):
                            adjusted_config_dict, adjusted_pes_list = helper_heuristics.readjust_pes_and_workloads(
                                mapping, mapping_config_dict)
                            adj_mapping = custom_mapping_utils.custom_mapping_from_desc_dict(board_name, model_dag,
                                                                                             adjusted_config_dict,
                                                                                             timing_metric, adjust_pes=False,
                                                                                             adjusted_pes_list=adjusted_pes_list)
                            perf_record = mapping_exec_utils.run_mapping(
                                board_name, model_name, adj_mapping)
                            adjusted_mapping_val = perf_record.get_metric_val(
                                opt_metric)
                            if opt_val_adjusted == -1 or first_better_than_second(opt_metric, adjusted_mapping_val, opt_val_adjusted):
                                opt_val_adjusted = adjusted_mapping_val
                                opt_mapping_dict_adjusted = copy.copy(
                                    adjusted_config_dict)

                        opt_val = current_val
                        opt_mapping_dict = copy.copy(mapping_config_dict)

        if opt_val_adjusted != -1 and first_better_than_second(opt_metric, opt_val_adjusted, opt_val):
            opt_mapping_dict = copy.copy(opt_mapping_dict_adjusted)
            opt_val = opt_val_adjusted
    ##################################################################################################################################
    # pe clustering
    if opt_metric != Metrics.LATENCY:
        features_mask_dict = helper_heuristics.power_of_2_clustering_mask_dict.keys()
        for mak_key in features_mask_dict:
            for num_clusters in range(constants.MIN_CLUSTERS, max_clusters + 1):
                cluster_layer_count_map, ordered_engines_pes = helper_heuristics.power_of_2_clustering(
                    model_dag, num_clusters, hw_config.num_pes, mak_key)
                if empty_clusters(cluster_layer_count_map):
                    break

                num_layers_so_far = 0
                num_engines_so_far = 0
                mapping_config_dict = {}
                for cluster_index in range(num_clusters):
                    current_cluster_num_layers = cluster_layer_count_map[cluster_index]
                    last_layer_offset = num_layers_so_far + current_cluster_num_layers
                    if current_cluster_num_layers == 1:
                        mapping_config_dict["{}".format(num_layers_so_far,
                                                        last_layer_offset)] = "{}".format(num_engines_so_far)
                    else:
                        mapping_config_dict["{}-{}".format(num_layers_so_far,
                                                           last_layer_offset)] = "{}".format(num_engines_so_far)

                    num_layers_so_far += current_cluster_num_layers
                    num_engines_so_far += 1
                mapping = custom_mapping_utils.custom_mapping_from_desc_dict(board_name, model_dag,
                                                                             mapping_config_dict, timing_metric, adjust_pes=False,
                                                                             adjusted_pes_list=ordered_engines_pes)

                perf_record = mapping_exec_utils.run_mapping(
                    board_name, model_name, mapping)
                current_val = perf_record.get_metric_val(opt_metric)
                if opt_val == -1 or first_better_than_second(opt_metric, current_val, opt_val):
                    opt_val = current_val
                    opt_mapping_dict = copy.copy(mapping_config_dict)

                #####################################
                current_val, new_mapping_config_dict, new_cluster_layer_counts, new_ordered_pes, mapping = adjust_with_the_unused_pes(
                    mapping, model_dag, ordered_engines_pes, hw_config,
                    cluster_layer_count_map,
                    board_name, model_name,
                    opt_metric, timing_metric)

                if opt_val == -1 or first_better_than_second(opt_metric, current_val, opt_val):
                    opt_val = current_val
                    opt_mapping_dict = copy.copy(new_mapping_config_dict)

                current_val, new_mapping_config_dict, new_cluster_layer_counts, new_ordered_pes = adjust_by_shifting(
                    mapping, model_dag, new_ordered_pes, hw_config,
                    new_cluster_layer_counts,
                    board_name, model_name,
                    opt_metric, timing_metric)

                if opt_val == -1 or first_better_than_second(opt_metric, current_val, opt_val):
                    opt_val = current_val
                    opt_mapping_dict = copy.copy(new_mapping_config_dict)

                #####################################
                adj_mapping = custom_mapping_utils.custom_mapping_from_desc_dict(board_name, model_dag,
                                                                                 mapping_config_dict,
                                                                                 timing_metric, adjust_pes=True)

                perf_record = mapping_exec_utils.run_mapping(
                    board_name, model_name, adj_mapping)
                adjusted_mapping_val = perf_record.get_metric_val(opt_metric)
                if opt_val_adjusted == -1 or first_better_than_second(opt_metric, adjusted_mapping_val, opt_val_adjusted):
                    opt_val_adjusted = adjusted_mapping_val
                    opt_mapping_dict_adjusted = copy.copy(mapping_config_dict)

        if opt_val_adjusted != -1 and first_better_than_second(opt_metric, opt_val_adjusted, opt_val):
            opt_mapping_dict = copy.copy(opt_mapping_dict_adjusted)
            opt_val = opt_val_adjusted
        #############################################################################################################
        # layer clustering
        for i in range(1):
            if i == 1 and not utils.has_dw_layers(model_dag):
                continue
            for num_clusters in range(constants.MIN_CLUSTERS, 5):
                cluster_layer_count_map = helper_heuristics.cluster_layers_into_close_powers_of_2(
                    model_dag, num_clusters, exclude_dw=(i == 1))
                if empty_clusters(cluster_layer_count_map):
                    break

                num_layers_so_far = 0
                num_engines_so_far = 0
                mapping_config_dict = {}
                for cluster_index in range(num_clusters):
                    current_cluster_num_layers = cluster_layer_count_map[cluster_index]
                    last_layer_offset = num_layers_so_far + current_cluster_num_layers
                    if current_cluster_num_layers == 1:
                        mapping_config_dict["{}".format(num_layers_so_far,
                                                        last_layer_offset)] = "{}".format(num_engines_so_far)
                    else:
                        mapping_config_dict["{}-{}".format(num_layers_so_far,
                                                           last_layer_offset)] = "{}".format(num_engines_so_far)

                    num_layers_so_far += current_cluster_num_layers
                    num_engines_so_far += 1
                mapping = custom_mapping_utils.custom_mapping_from_desc_dict(board_name, model_dag,
                                                                             mapping_config_dict, timing_metric, adjust_pes=True)

                perf_record = mapping_exec_utils.run_mapping(
                    board_name, model_name, mapping)
                current_val = perf_record.get_metric_val(opt_metric)
                if opt_val == -1 or first_better_than_second(opt_metric, current_val, opt_val):
                    opt_val = current_val
                    opt_mapping_dict = copy.copy(mapping_config_dict)

                adjusted_config_dict, adjusted_pes_list = helper_heuristics.readjust_pes_and_workloads(
                    mapping, mapping_config_dict)
                adj_mapping = custom_mapping_utils.custom_mapping_from_desc_dict(board_name, model_dag,
                                                                                 mapping_config_dict,
                                                                                 timing_metric, adjust_pes=False,
                                                                                 adjusted_pes_list=adjusted_pes_list)

                perf_record = mapping_exec_utils.run_mapping(
                    board_name, model_name, adj_mapping)
                adjusted_mapping_val = perf_record.get_metric_val(opt_metric)
                if opt_val_adjusted == -1 or first_better_than_second(opt_metric, adjusted_mapping_val, opt_val_adjusted):
                    opt_val_adjusted = adjusted_mapping_val
                    opt_mapping_dict_adjusted = copy.copy(mapping_config_dict)

                current_val, new_mapping_config_dict, new_cluster_layer_counts, new_ordered_pes, mapping = adjust_with_the_unused_pes(
                    adj_mapping, model_dag, adjusted_pes_list, hw_config,
                    cluster_layer_count_map,
                    board_name, model_name,
                    opt_metric, timing_metric)

                if opt_val == -1 or first_better_than_second(opt_metric, current_val, opt_val):
                    opt_val = current_val
                    opt_mapping_dict = copy.copy(new_mapping_config_dict)

                current_val, new_mapping_config_dict, new_cluster_layer_counts, new_ordered_pes = adjust_by_shifting(
                    mapping, model_dag, new_ordered_pes, hw_config,
                    new_cluster_layer_counts,
                    board_name, model_name,
                    opt_metric, timing_metric)
                
                if opt_val == -1 or first_better_than_second(opt_metric, current_val, opt_val):
                    opt_val = current_val
                    opt_mapping_dict = copy.copy(new_mapping_config_dict)

            if opt_val_adjusted != -1 and first_better_than_second(opt_metric, opt_val_adjusted, opt_val):
                opt_mapping_dict = copy.copy(opt_mapping_dict_adjusted)
                opt_val = opt_val_adjusted

    print(opt_val, opt_mapping_dict)
    duration = time.time() - t0

    timing_file_name = 'hm_max_clusters_{}.json'.format(max_clusters)
    timings_dict = mapping_general_utils.load_json_to_dict(
        os.path.dirname(__file__) + '/timing/' + timing_file_name)

    if timings_dict != None:
        new_timing = duration
        if board_name in timings_dict:
            if model_name in timings_dict[board_name]:
                previous_runs = int(
                    timings_dict[board_name][model_name]['runs'])
                previous_timing = float(
                    timings_dict[board_name][model_name]['duration'])
                previous_timing *= previous_runs
                new_timing = (previous_timing + duration) / (1 + previous_runs)
            else:
                timings_dict[board_name][model_name] = {}
        else:
            timings_dict[board_name] = {model_name: {}}
        if 'runs' not in timings_dict[board_name][model_name]:
            timings_dict[board_name][model_name]['runs'] = 0
        timings_dict[board_name][model_name]['runs'] += 1
        timings_dict[board_name][model_name]['duration'] = new_timing
    else:
        timings_dict = {}
        timings_dict[board_name] = {model_name: {}}
        timings_dict[board_name][model_name]['runs'] = 1
        timings_dict[board_name][model_name]['duration'] = duration

    mapping_general_utils.save_dict_to_json(timings_dict, os.path.dirname(__file__) + '/timing/',
                                            timing_file_name)

    return opt_val, opt_mapping_dict
