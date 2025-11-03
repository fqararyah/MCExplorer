import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import utils
from mapping_utils import mapping_general_utils
from mapping_utils import helper_heuristics
import copy
import itertools
from time import sleep
import random

def get_effective_pes(initial_pes):
    effective_pes_list = []

    for budget in initial_pes:
        effective_pes_list.append(utils.pow_of_2_leq(budget))

    return effective_pes_list


def get_pes_to_reach_next_pow_of_2(effective_pe_list, assigned_pe_list):
    to_reach_pe_list = []
    for i in range(len(effective_pe_list)):
        to_reach_pe_list.append(effective_pe_list[i] * 2 - assigned_pe_list[i])

    return to_reach_pe_list


def get_extra_pes(effective_pe_list, assigned_pe_list):
    extra_pes_list = []

    for i in range(len(effective_pe_list)):
        extra_pes = assigned_pe_list[i] - effective_pe_list[i]
        extra_pes_list.append(extra_pes)

    return extra_pes_list


def get_wasted_pes_proportional(extra_pes_list, effective_pes_list, num_segments):
    wasted_pes_proportional_list = [0] * num_segments
    for i in range(num_segments):
        wasted_pes_proportional_list[i] = extra_pes_list[i] / \
            effective_pes_list[i]

    return wasted_pes_proportional_list


def readjust_pes(assigned_pes_list, total_pes):
    adjusted_pes_list = copy.copy(assigned_pes_list)
    unused_pes = total_pes - sum(adjusted_pes_list)
    num_segments = len(adjusted_pes_list)
    effective_pes_list = get_effective_pes(adjusted_pes_list)
    extra_pes_list = get_extra_pes(effective_pes_list, adjusted_pes_list)
    sum_extra_pes = sum(extra_pes_list)

    max_waste_to_reach_pes = 1
    to_rech_pe_list = get_pes_to_reach_next_pow_of_2(
        effective_pes_list, adjusted_pes_list)
    wasted_pes_proportional_list = get_wasted_pes_proportional(
        extra_pes_list, effective_pes_list, num_segments)

    while sum_extra_pes > 0 and max_waste_to_reach_pes > 0:
        max_wasted_pes_proportional = max(wasted_pes_proportional_list)
        max_wasted_pes_proportional_index = wasted_pes_proportional_list.index(
            max_wasted_pes_proportional)
        wasted_pes_proportional_list[max_wasted_pes_proportional_index] = 0
        extra_pes_list[max_wasted_pes_proportional_index] = 0
        max_waste_extra_pes = extra_pes_list[max_wasted_pes_proportional_index]
        max_waste_to_reach_pes = to_rech_pe_list[max_wasted_pes_proportional_index]
        if max_waste_to_reach_pes > (sum_extra_pes + unused_pes):
            to_rech_pe_list[max_wasted_pes_proportional_index] = 0
            adjusted_pes_list[max_wasted_pes_proportional_index] -= max_waste_extra_pes
            continue

        to_rech_pe_list[max_wasted_pes_proportional_index] = 0
        adjusted_pes_list[max_wasted_pes_proportional_index] += max_waste_to_reach_pes
        effective_pes_list[max_wasted_pes_proportional_index] = adjusted_pes_list[max_wasted_pes_proportional_index]

        if unused_pes >= max_waste_to_reach_pes:
            unused_pes -= max_waste_to_reach_pes
        else:
            unused_pes = 0

        while max_waste_to_reach_pes > 0 and sum(extra_pes_list) > 0:
            max_extra_pes = max(extra_pes_list)
            max_extra_pes_index = extra_pes_list.index(max_extra_pes)
            if max_extra_pes >= max_waste_to_reach_pes:
                extra_pes_list[max_extra_pes_index] -= max_waste_to_reach_pes
                adjusted_pes_list[max_extra_pes_index] -= max_waste_to_reach_pes
                wasted_pes_proportional_list[max_extra_pes_index] -= max_waste_extra_pes / \
                    effective_pes_list[max_extra_pes_index]
                to_rech_pe_list[max_extra_pes_index] += max_waste_to_reach_pes
                sum_extra_pes -= max_waste_to_reach_pes
            else:
                adjusted_pes_list[max_extra_pes_index] = effective_pes_list[max_extra_pes_index]
                extra_pes_list[max_extra_pes_index] = 0
                wasted_pes_proportional_list[max_extra_pes_index] = 0
                to_rech_pe_list[max_extra_pes_index] = 0
                sum_extra_pes -= max_extra_pes

    sum_extra_pes = total_pes - sum(adjusted_pes_list)

    for i in range(num_segments):
        if sum_extra_pes >= adjusted_pes_list[i]:
            sum_extra_pes -= adjusted_pes_list[i]
            assigned_pes_list[i] *= 2

    sum_extra_pes_by_segments = sum_extra_pes // num_segments
    reminder_pes = sum_extra_pes % num_segments
    for i in range(num_segments):
        adjusted_pes_list[i] += sum_extra_pes_by_segments
        if reminder_pes > 0:
            adjusted_pes_list[i] += 1
            reminder_pes -= 1

    return adjusted_pes_list


def decompose_into_sum_of_powers_of_two(num, num_powers_of_two = -1, least_permissible=1):
    total_pes_binary_list = mapping_general_utils.dec_to_binary_as_list(
        num)
    num_ones = sum(total_pes_binary_list)
    index_in_list = 0
    if num_powers_of_two <= -1:
        num_powers_of_two = num_ones

    while num_ones > num_powers_of_two:
        # if there are more ones, remove the least significant
        if total_pes_binary_list[index_in_list] == 1:
            num_ones -= 1
        total_pes_binary_list[index_in_list] = 0
        index_in_list += 1

    pow_of_2_list = []
    for i in range(len(total_pes_binary_list)):
        if total_pes_binary_list[i] == 1:
            pes_to_add = 2 ** i
            if pes_to_add < least_permissible:
                total_pes_binary_list[i] = 0
            else:
                pow_of_2_list.append(pes_to_add)

    assert len(pow_of_2_list) <= num_powers_of_two

    pow_of_2_list.sort(reverse=True)
    while len(pow_of_2_list) < num_powers_of_two and pow_of_2_list[0] >= least_permissible * 2:
        to_add = pow_of_2_list[0] / 2
        pow_of_2_list[0] /= 2
        pow_of_2_list.append(to_add)
        pow_of_2_list.sort(reverse=True)

    return pow_of_2_list


def readjust_pes_v2(assigned_pes_list, total_pes, min_cluster_allowed_pes):
    num_clusters = len(assigned_pes_list)
    total_pes_binary_list = mapping_general_utils.dec_to_binary_as_list(
        total_pes)
    num_ones = sum(total_pes_binary_list)
    index_in_list = 0
    while num_ones > num_clusters:
        if total_pes_binary_list[index_in_list] == 1:
            num_ones -= 1
        total_pes_binary_list[index_in_list] = 0
        index_in_list += 1

    pow_of_2_list = []
    for i in range(len(total_pes_binary_list)):
        if total_pes_binary_list[i] == 1:
            pes_to_add = 2 ** i
            if pes_to_add < min_cluster_allowed_pes:
                total_pes_binary_list[i] = 0
                num_ones -= 1
            else:
                pow_of_2_list.append(pes_to_add)

    assert len(pow_of_2_list) <= num_clusters

    assigned_pes_list_indices = [i for i in range(num_clusters)]
    assigned_pes_list_sorted, assigned_pes_list_sorted_indices = zip(
        *sorted(zip(assigned_pes_list, assigned_pes_list_indices)))
    adjusted_pes_list = []
    if num_ones == num_clusters:
        for i in range(num_clusters):
            adjusted_pes_list.append(
                pow_of_2_list[assigned_pes_list_sorted_indices[i]])
        return adjusted_pes_list
    else:
        pow_of_2_list.sort()
        diffs = []
        while len(pow_of_2_list) < num_clusters:
            for i in range(len(pow_of_2_list)):
                diffs.append(pow_of_2_list[i] - assigned_pes_list_sorted[i])
            max_index = diffs.index(max(diffs))
            if pow_of_2_list[max_index] < 2 * min_cluster_allowed_pes:
                max_index = pow_of_2_list.index(max(pow_of_2_list))
            pow_of_2_list[max_index] /= 2
            pow_of_2_list.append(pow_of_2_list[max_index])

    for i in range(num_clusters):
        adjusted_pes_list.append(
            pow_of_2_list[assigned_pes_list_sorted_indices[i]])

    sum_extra_pes = total_pes - sum(adjusted_pes_list)
    assert sum_extra_pes >= 0

    for i in range(num_clusters):
        if sum_extra_pes >= adjusted_pes_list[i]:
            sum_extra_pes -= adjusted_pes_list[i]
            assigned_pes_list[i] *= 2

    sum_extra_pes_by_segments = sum_extra_pes // num_clusters
    reminder_pes = sum_extra_pes % num_clusters
    for i in range(num_clusters):
        adjusted_pes_list[i] += sum_extra_pes_by_segments
        if reminder_pes > 0:
            adjusted_pes_list[i] += 1
            reminder_pes -= 1

    return adjusted_pes_list


def reassign_workloads(op_counts_list, adjusted_pes_list):
    num_clusters = len(adjusted_pes_list)
    total_ops = sum(op_counts_list)
    total_pes = sum(adjusted_pes_list)
    max_pes_index = adjusted_pes_list.index(max(adjusted_pes_list))
    adjusted_clusters = []

    ops_ratio_so_far = 0
    pes_ratio_so_far = adjusted_pes_list[0] / total_pes
    abs_index = 0
    for i in range(num_clusters):
        current_index_in_cluster = 0
        adjusted_clusters.append([])
        while ops_ratio_so_far < pes_ratio_so_far and abs_index < len(op_counts_list):
            current_ops_ratio = op_counts_list[abs_index] / total_ops
            # i == max_pes_index, the custer with largest PEs is the one that is mose tolerant (overhead is proportionally lower)
            if current_index_in_cluster == 0 or ops_ratio_so_far + current_ops_ratio <= pes_ratio_so_far or i == max_pes_index:
                adjusted_clusters[-1].append(op_counts_list[abs_index])
                ops_ratio_so_far += current_ops_ratio
                current_index_in_cluster += 1
                abs_index += 1
            else:
                break

        if i < num_clusters - 1:
            pes_ratio_so_far += adjusted_pes_list[i + 1] / total_pes

    return adjusted_clusters


def readjust_pes_and_workloads(mapping, mapping_dict, adjustment_heuristic_index=0):
    model_dag = mapping.model_dag
    total_pes = mapping.hw_config.num_pes

    op_counts_list = utils.get_layers_op_counts(model_dag, 0)
    pes_list = mapping.get_sub_mappings_pes()
    min_cluster_allowed_pes = total_pes * \
        (max(op_counts_list) / sum(op_counts_list))
    adjusted_mapping_dict = {}
    adjusted_pes_list = []
    if adjustment_heuristic_index == 0:
        adjusted_pes_list = readjust_pes_v2(
            pes_list, total_pes, min_cluster_allowed_pes)
        adjusted_clusters = reassign_workloads(
            op_counts_list, adjusted_pes_list)
        adjusted_mapping_dict = mapping_general_utils.adjust_mapping_dict_layers(
            mapping_dict, adjusted_clusters)

    return adjusted_mapping_dict, adjusted_pes_list


def extract_layer_features(layer_specs):
    feature = utils.get_layer_ofms_shape(layer_specs)[1:3]
    feature.append(utils.get_layer_index(layer_specs))

    return feature


features_mask_dict = {
    0: [1, 0, 0, 0],
    0: [0, 1, 0, 0],
    1: [0, 0, 1, 0],
    2: [0, 0, 0, 1]
}


def extract_layer_features_v2(layer_specs, feature_mask):

    ifms_size = utils.get_layer_ifms_size(layer_specs)
    ofms_size = utils.get_layer_ofms_size(layer_specs)
    filter_shape = utils.get_layer_weights_shape(layer_specs)
    ifms_shape = utils.get_layer_ifms_shape(layer_specs)
    ofms_shape = utils.get_layer_ofms_shape(layer_specs)

    fms_size = ifms_size + ofms_size
    filter_ind_size = filter_shape[0] * filter_shape[1]
    filters_to_fms_size_ratio = filter_ind_size / fms_size
    ifms_filters_d = ifms_shape[0]  # + ofms_shape[2]
    ifms_hw = ifms_shape[1] + ifms_shape[2]  # + ofms_shape[2]
    ofms_hw = ofms_shape[1] + ofms_shape[2]
    feature_list = [ifms_filters_d,
                    filters_to_fms_size_ratio, ifms_hw, ofms_hw]

    for i, desired_feature in enumerate(feature_mask):
        feature_list[i] *= desired_feature

    return feature_list


def extract_features(model_dag, features_version, feature_mask_key=0):
    num_conv_layers = utils.get_num_conv_layer_count_in_range(
        model_dag, 0, len(model_dag))
    if features_version == 1:
        features_arr = np.zeros([num_conv_layers, 3])
    else:
        features_arr = np.zeros([num_conv_layers, len(features_mask_dict[0])])

    layer_index = 0
    first_layer_unique = utils.has_dw_layers(model_dag)
    for layer_specs in model_dag:
        if utils.is_conv_layer(layer_specs):
            if features_version == 1:
                features_arr[layer_index] = extract_layer_features(layer_specs)
            else:
                feature_mask = features_mask_dict[feature_mask_key]
                features_arr[layer_index] = extract_layer_features_v2(
                    layer_specs,  feature_mask)
            layer_index += 1

    scaled_features_arr = features_arr  # scaler.fit_transform(features_arr)

    return scaled_features_arr


def cluster_layers_using_kmeans(model_dag, num_clusters, max_iter=300, features_version=1, feature_mask_key=0):
    features_arr = extract_features(
        model_dag, features_version=features_version, feature_mask_key=feature_mask_key)
    kmeans = KMeans(n_clusters=num_clusters, random_state=1,
                    max_iter=max_iter).fit(features_arr)
    silhouette_score_val = silhouette_score(features_arr, kmeans.labels_)
    cluster_layers_map = {}
    cluster_index_map = {}
    cluster_layer_count_map = [0] * num_clusters
    layer_index = 0
    feature_mask = features_mask_dict[feature_mask_key]
    for layer_specs in model_dag:
        if utils.is_conv_layer(layer_specs):
            layer_features = extract_layer_features(layer_specs) if features_version == 1 else \
                extract_layer_features_v2(
                    layer_specs, feature_mask)
            predicted_cluster = kmeans.predict(
                [layer_features])[0]
            if predicted_cluster not in cluster_layers_map:
                cluster_layers_map[predicted_cluster] = []
                cluster_index_map[predicted_cluster] = len(cluster_index_map)

            cluster_layer_count_map[cluster_index_map[predicted_cluster]] += 1
            cluster_layers_map[predicted_cluster].append(
                utils.get_layer_index(layer_specs))
            layer_index += 1

    return cluster_layer_count_map, silhouette_score_val


def minimize_max_ratio_exh(lists, total_pes):
    num_engines = len(lists)

    max_list_len = max(len(current_list) for current_list in lists)
    op_lists_sums_list = []
    for current_list in lists:
        op_lists_sums_list.append(sum(current_list))
        for i in range(len(current_list), max_list_len):
            current_list.append(0)

    sum_op_lists_sums_list = sum(op_lists_sums_list)
    engine_loads_ratios_list = [op_lists_sum / sum_op_lists_sums_list for op_lists_sum in op_lists_sums_list]
    # Initialize variables to track the best result
    best_vars = [0] * num_engines
    best_value = float('inf')

    # Generate all possible combinations of powers of 2
    powers_of_2 = [2**i for i in range(int(np.log2(total_pes)) + 1)]
    combinations = itertools.product(powers_of_2, repeat=num_engines)

    for comb in combinations:
        if sum(comb) <= total_pes:
            # Calculate the sum of the max ratios for each list
            current_value = sum(max(
                lists[k][i] / comb[k] for k in range(num_engines)) for i in range(len(lists[0])))
            for k in range(num_engines):
                current_value += comb[k]
            if current_value < best_value:
                best_value = current_value
                best_vars = comb

    unassigned_pes = total_pes - sum(best_vars)
    best_vars = list(best_vars)
    for i in range(len(best_vars)):
        best_vars[i] += int(engine_loads_ratios_list[i] * unassigned_pes)

    return best_vars, best_value


def minimize_max_ratio_greedy(lists, total_pes, segment_engine_activity):
    num_engines = len(lists)

    max_list_len = max(len(current_list) for current_list in lists)
    # for current_list in lists:
    #     for i in range(len(current_list), max_list_len):
    #         current_list.append(0)

    # Initialize variables to track the best result
    # Initialize with ones to avoid division by zero
    best_vars = [1] * num_engines

    # Distribute PEs using a greedy approach
    used_pes = num_engines
    saturated_engines = [0] * num_engines
    while used_pes < total_pes:  # Adjust for initial allocation of 1 PE to each engine
        slowest_engine = -1
        per_stage_durations = []
        max_durations = [0] * num_engines
        engine_current_layer_indices_list = [0] * num_engines
        for k in range(num_engines):
            for stage_index in range(max_list_len):
                if k == 0:
                    per_stage_durations.append([])
                if saturated_engines[k] == 1 or segment_engine_activity[stage_index][k] == 0:
                    per_stage_durations[stage_index].append(0)
                else:
                    per_stage_durations[stage_index].append(
                        lists[k][engine_current_layer_indices_list[k]] / best_vars[k])
                    engine_current_layer_indices_list[k] += 1
        for stage_index in range(max_list_len):
            max_duration = max(per_stage_durations[stage_index])
            max_duration_index = per_stage_durations[stage_index].index(
                max_duration)
            max_durations[max_duration_index] += max_duration

        slowest_engine = max_durations.index(max(max_durations))
        if used_pes + best_vars[slowest_engine] <= total_pes:
            used_pes += best_vars[slowest_engine]
            best_vars[slowest_engine] *= 2
        else:
            saturated_engines[slowest_engine] = 1
            if sum(saturated_engines) == num_engines or max_duration == 0:
                break
    
    return best_vars


def canDistributeItems(items, bags, maxOverflow):
    bag_index = 0
    current_weight = 0
    current_bag_items = 0
    bag_distributions = [[] for _ in bags]

    for index, item in enumerate(items):
        # If adding the item would exceed the bag capacity + max overflow
        if current_weight + item > bags[bag_index] + maxOverflow:
            # Move to the next bag
            bag_distributions[bag_index] = current_bag_items
            bag_index += 1
            current_weight = item
            current_bag_items = 1
            if bag_index >= len(bags):  # If no more bags are available
                return False, []
        else:
            current_weight += item  # Add item to the current bag
            current_bag_items.append(index)

    # Store the last set of items in the last bag
    bag_distributions[bag_index] = current_bag_items

    return True, bag_distributions


def canDistributeItems(items, bags, maxOverflow):
    bag_index = 0
    current_weight = 0
    current_bag_items = []
    bag_distributions = [[] for _ in bags]

    for item in items:
        # If adding the item would exceed the bag capacity + max overflow
        if current_weight + item > bags[bag_index] + maxOverflow:
            # Move to the next bag
            bag_distributions[bag_index] = current_bag_items
            bag_index += 1
            current_weight = item
            current_bag_items = [item]
            if bag_index >= len(bags):  # If no more bags are available
                return False, []
        else:
            current_weight += item  # Add item to the current bag
            current_bag_items.append(item)

    # Store the last set of items in the last bag
    bag_distributions[bag_index] = current_bag_items

    return True, bag_distributions

def canDistributeItems_relative(items, bags, maxRelativeOverflow):
    bag_index = 0
    current_weight = 0
    current_bag_items = []
    bag_distributions = [[] for _ in bags]

    for item in items:
        # If adding the item would exceed the bag's relative capacity (max relative overflow)
        if current_weight + item > bags[bag_index] * (1 + maxRelativeOverflow):
            # Move to the next bag
            bag_distributions[bag_index] = current_bag_items
            bag_index += 1
            current_weight = item
            current_bag_items = [item]
            if bag_index >= len(bags):  # If no more bags are available
                return False, []
        else:
            current_weight += item  # Add item to the current bag
            current_bag_items.append(item)

    # Store the last set of items in the last bag
    bag_distributions[bag_index] = current_bag_items

    return True, bag_distributions

def minimize_overflow_permutations(items, bags, relative):
    best_distribution = None
    best_max_overflow = float('inf')

    best_distribution_across_all = None
    best_max_overflow_across_all = float('inf')
    best_order = None
    # Try all permutations of the bags to account for different bag orderings
    bag_indices = [i for i in range(len(bags))]
    for bag_order in itertools.permutations(bag_indices):
        left = 0  # Lower bound of overflow (no overflow)
        if relative:
            right = max(items)
        else:
            right = sum(items)
        ordered_bags = [bags[bag_order[i]] for i in range(len(bags))]
        while left < right:
            if relative:
                mid = (left + right) / 2  # Candidate for max relative overflow
            else:
                mid = (left + right) // 2  # Candidate for max relative overflow

            feasible, distribution = canDistributeItems(
                items, ordered_bags, mid)
            if feasible:
                best_distribution = distribution  # Store the valid distribution
                best_max_overflow = mid
                if best_max_overflow < best_max_overflow_across_all:
                    best_max_overflow_across_all = best_max_overflow
                    best_distribution_across_all = copy.copy(best_distribution)
                    best_order = copy.copy(bag_order)
                if relative:
                    right = mid - 0.01  # Try for a smaller relative overflow
                else:
                    right = mid - 1
            else:
                if relative:
                    left = mid + 0.01  # Increase the allowed relative overflow
                else:
                    left = mid + 1 

    return best_max_overflow_across_all, best_distribution_across_all, best_order


def minimize_overflow_greedy(items, bags, relative):
    left = 0  # Lower bound of relative overflow
    # Upper bound of relative overflow (large overflows possible)
    if relative:
        right = max(items)
    else:
        right = sum(items)
    best_max_relative_overflow = right
    best_distribution = None

    while left <= right:
        if relative:
            mid = (left + right) / 2  # Candidate for max relative overflow
        else:
            mid = (left + right) // 2  # Candidate for max relative overflow

        feasible, distribution = canDistributeItems(items, bags, mid)

        if feasible:
            best_max_relative_overflow = mid
            best_distribution = distribution
            if relative:
                right = mid - 0.01  # Try for a smaller relative overflow
            else:
                right = mid - 1
        else:
            if relative:
                left = mid + 0.01  # Increase the allowed relative overflow
            else:
                left = mid + 1 

    return best_max_relative_overflow, best_distribution

def homo_powers_of_two(num, num_powers):
    avg = num / num_powers
    pow_of_2 = utils.pow_of_2_leq(avg)
    powers_of_2_list = [pow_of_2] * num_powers
    powers_sum = sum(powers_of_2_list)
    unused = num - powers_sum
    i = 0
    while unused >= powers_of_2_list[i]:
        unused -= powers_of_2_list[i]
        powers_of_2_list[i] *= 2

        i += 1

    for i in range(1, num_powers):
        if unused + powers_of_2_list[i] // 2 >= powers_of_2_list[i - 1]:
            powers_of_2_list[i - 1] *= 2
            unused -= powers_of_2_list[i] // 2
            powers_of_2_list[i] //= 2

    assert sum(powers_of_2_list) <= num
    random.shuffle(powers_of_2_list)

    return powers_of_2_list

power_of_2_clustering_mask_dict = {
    0: [0, 0],
    0: [0, 1],
    1: [1, 0],
    2: [1, 1]
}


def closest_power_of_two(value):
    if value == 0:
        return 0
    return 2 ** (math.floor(math.log2(value)))

# Function to calculate the difference from the closest power of two
def power_of_two_diff(sublist_sum, comparison_sum):
    ratio = sublist_sum / comparison_sum if comparison_sum != 0 else float('inf')
    return abs(ratio - closest_power_of_two(ratio))

# Function to partition the list into k contiguous sublists
def cluster_layers_into_close_powers_of_2(model_dag, k, exclude_dw = False):
    op_count_list = utils.get_layers_op_counts(model_dag, 0)
    if exclude_dw and utils.has_dw_layers(model_dag):
        for i in range(len(op_count_list)):
            layer_index = utils.get_conv_layer_index_from_offset(model_dag, 0, i)
            layer_specs = model_dag[layer_index]
            if layer_specs['type'] == 'dw':
                op_count_list[i] = 0

    n = len(op_count_list)
    total_sum = sum(op_count_list)

    # If it's impossible to split into exactly k sublists, return None
    if k > n:
        return None

    best_partition = None
    min_diff = float('inf')
    
    # Generate all possible partitions into exactly k contiguous sublists
    cuts = [i for i in range(1, n - 1)]  # Possible cut positions between 0 and n-1
    for cut_points in itertools.combinations(cuts, k-1):
        # Add the first and last cut points to make partitions
        cut_points = (0, *cut_points, n)

        sublists = [op_count_list[cut_points[i] : cut_points[i + 1]] for i in range(k)]

        # Ensure that the sum of the sublist sums equals the sum of nums
        sublist_sums = [sum(sublist) for sublist in sublists]
        assert sum(sublist_sums) == total_sum
            
        # Calculate the total difference from powers of 2 for this partition
        total_closeness = 0
        for i in range(k):
            if sublist_sums[i] == 0:
                break
            for j in range(i + 1, k):
                if sublist_sums[j] == 0:
                    break
                total_closeness += power_of_two_diff(sublist_sums[i], sublist_sums[j])

        # If the current partition is better, store it as the best partition
        if total_closeness < min_diff:
            min_diff = total_closeness
            best_partition = sublists

    sum_sums = 0
    clusters_layers = []
    for sub_list in best_partition:
        sum_sums += sum(sub_list)
        clusters_layers.append(len(sub_list))

    assert sum_sums == total_sum

    return clusters_layers

def power_of_2_clustering(model_dag, num_engines, num_pes,options_key):
    orig_num_pes = num_pes
    options_list = power_of_2_clustering_mask_dict[options_key]
    dw_ops_sum = 0
    op_count_list = utils.get_layers_op_counts(model_dag, 0)
    if utils.has_dw_layers(model_dag):
        for i in range(len(op_count_list)):
            layer_index = utils.get_conv_layer_index_from_offset(model_dag, 0, i)
            layer_specs = model_dag[layer_index]
            if layer_specs['type'] == 'dw':
                dw_ops_sum += op_count_list[i]
                op_count_list[i] = 0

        pw_s_ops_sum = sum(op_count_list)
        dw_pes = (num_pes * dw_ops_sum) // (pw_s_ops_sum + dw_ops_sum)
        num_pes -= dw_pes

    if options_list[0] == 0:
        engines_pes = decompose_into_sum_of_powers_of_two(
            num_pes, num_engines, least_permissible=-1)
    else:
        engines_pes = homo_powers_of_two(num_pes, num_engines)

    sum_ops = sum(op_count_list)
    sum_pes = sum(engines_pes)

    if num_engines < 8:
        random.shuffle(engines_pes)
    scaled_engines_pes = [int(engines_pes[i] * sum_ops / sum_pes)
                          for i in range(num_engines)]
    if num_engines < 8:  # >= 8! is large, and the difference is not that great anyway
        _, clusters, order = minimize_overflow_permutations(
            op_count_list, scaled_engines_pes, options_list[1])
        ordered_engines_pes = [engines_pes[order[i]]
                               for i in range(len(engines_pes))]
        scaled_engines_pes = [scaled_engines_pes[order[i]]
                               for i in range(len(scaled_engines_pes))]
    else:
        _, clusters = minimize_overflow_greedy(
            op_count_list, scaled_engines_pes, options_list[1])
        ordered_engines_pes = engines_pes

    cluster_sums = [sum(clusters[i]) for i in range(len(clusters))]
    sum_sums = sum(cluster_sums)
    assert sum_sums == sum_ops

    if utils.has_dw_layers(model_dag):
        dw_pes += num_pes - sum(ordered_engines_pes) #add unused pes
        op_count_list = utils.get_layers_op_counts(model_dag, 0)
        abs_index = 0
        for cluster_index, cluster in enumerate(clusters):
            sum_dw_ops_in_cluster = 0
            for i in range(len(cluster)):
                sum_dw_ops_in_cluster += op_count_list[abs_index] - cluster[i]
                abs_index += 1
            proportional_dw_pes = (dw_pes * sum_dw_ops_in_cluster) // dw_ops_sum
            ordered_engines_pes[cluster_index] += proportional_dw_pes

    assert sum(ordered_engines_pes) <= orig_num_pes

    cluster_layer_counts = [len(cluster) for cluster in clusters]

    return cluster_layer_counts, ordered_engines_pes
