from plot import plot_results
import experiments.experiments as exps exps
import constants as consts
from mapping_strategies.mapping_types.hybrid_mapping import *
from mapping_strategies.mapping_types.segment_grained_mapping_rr import *
from mapping_strategies.mapping_types.segment_grained_mapping import *
import os
import __init__
import utils
from preformance_record import *
import mapping_strategies.mapping_utils.mapping_general_utils as mapping_general_utils
import statistics


def plot_estimation_errors(errors, plot_name, board_name, x_ticks_dict, colors=[]):
    # plot_results.bar_model_groups(engine_list, [errors], 'errors', [model_name],
    #                               [x_axis_label, 'error'], plot_name, board_name, mapping, normalized=False, colors=colors)
    plot_results.plot_bar_groups(errors, plot_name,
                                 x_title=None, y_title='Error', x_ticks_dict=x_ticks_dict,
                                 relative_save_path='{}/errors/'.format(board_name), percentage=True)


def read_actual_results(board_name, mapping_name, metric_dir, min_engines, max_engines, final = True):
    act_results = {}
    results_dir = consts.CLEANED_RESULTS_DIR + '/' + \
        board_name + '_f/' if final else '' + mapping_name + '/' + metric_dir

    for file_name in os.listdir(results_dir):
        print('read actual:', file_name)
        model_name = file_name.split('.')[0]
        if model_name not in act_results:
            act_results[model_name] = {}
        file_path = results_dir + '/' + file_name
        with open(file_path, 'r') as f:
            line_num = 0
            if mapping_name == 'segrr' and 'latenc' in metric_dir:
                for line in f:
                    if line_num >= min_engines and line_num <= max_engines:
                        line = utils.clean_line(line)
                        splits = line.split(',')
                        act_results[model_name][line_num] = {}
                        for split in splits:
                            if split == '':
                                continue
                            sub_splits = split.split(':')
                            segment_index = int(sub_splits[0])
                            exec_time = float(sub_splits[1])
                            act_results[model_name][line_num][int(
                                segment_index)] = float(exec_time)
                    line_num += 1
            else:
                for line in f:
                    if line_num >= min_engines and line_num <= max_engines:
                        line = utils.clean_line(line)
                        act_results[model_name][line_num] = float(line)
                    line_num += 1

    return act_results


def calc_estimation_errors(mapping, actual_results_list, metric, min_engines, max_engines):
    errors = {}
    predicted_dict = {}
    signed_errors = {}

    if mapping == 'hyb':
        predictions_list = exps.engines_and_metrics_mapping(
            ['vcu108'], consts.model_names, min_engines, max_engines,
            mapping_labels=[HybridMapping.MAPPING_LABEL], plot=False, print_results=False)
    elif mapping == 'segrr':
        predictions_list = exps.engines_and_metrics_mapping(
            ['vcu108'], consts.model_names, min_engines, max_engines,
            mapping_labels=[SegmentMappingRR.MAPPING_LABEL], plot=False, print_results=False)
    elif mapping == 'seg':
        predictions_list = exps.engines_and_metrics_mapping(
            ['vcu108'], consts.model_names, min_engines, max_engines,
            mapping_labels=[SegmentMapping.MAPPING_LABEL], plot=False, print_results=False)

    for prediction in predictions_list:
        model_name = prediction.model_name
        num_engines = prediction.num_engines
        if model_name not in predicted_dict:
            predicted_dict[model_name] = []
        if model_name in actual_results_list:
            actual_result = actual_results_list[model_name][num_engines]
            if metric == Metrics.BUFFER:
                predicted_result = prediction.on_chip_buffer
                predicted_dict[model_name].append(predicted_result)
                actual_predicted_diff = abs(predicted_result - actual_result)
                actual_predicted_diff_signed = predicted_result - actual_result
                den = 1 + max(predicted_result, actual_result)
                if model_name == 'resnet152' and mapping == 'seg':
                    print(predicted_result, actual_result)
            elif metric == Metrics.LATENCY:
                predicted_dict[model_name].append(prediction.latency)
                if mapping == 'hyb':
                    predicted_result = prediction.latency
                    predicted_result /= 1000  # s to ms
                    den = 0.01 + max(predicted_result, actual_result)
                    actual_predicted_diff = abs(
                        actual_result - predicted_result)
                    actual_predicted_diff_signed = predicted_result - actual_result
                elif mapping == 'segrr':
                    den = 0.01  # avoid div by zero
                    actual_predicted_diff = 0
                    actual_predicted_diff_signed = 0
                    predicted_result = prediction.segment_exec_times
                    for segment_index, exec_time in actual_result.items():
                        # TODO: fix (remove)
                        if segment_index == 1 and model_name == 'mob_v2':
                            continue
                        actual_predicted_diff += abs(exec_time -
                                                     predicted_result[segment_index])
                        actual_predicted_diff_signed += predicted_result[segment_index] - exec_time
                        den += max(exec_time, predicted_result[segment_index])
            elif metric == Metrics.THROUGHPUT:
                predicted_dict[model_name].append(prediction.throughput)
                if mapping == 'hyb':
                    predicted_result = prediction.latency
                    predicted_result /= 1000  # s to ms
                    den = 0.01 + max(predicted_result, actual_result)
                    actual_predicted_diff = abs(
                        actual_result - predicted_result)
                    actual_predicted_diff_signed = actual_result - predicted_result
                elif mapping == 'segrr':
                    actual_predicted_diff = 0
                    actual_predicted_diff_signed = 0
                    predicted_result = prediction.segment_exec_times
                    predicted_max = 0
                    actual_max = 0
                    for segment_index, exec_time in actual_result.items():
                        # TODO: fix (remove)
                        if segment_index == 1 and model_name == 'mob_v2':
                            continue
                        predicted_max = max(
                            predicted_max, predicted_result[segment_index])
                        actual_max = max(actual_max, exec_time)

                        actual_predicted_diff += abs(predicted_max - actual_max)
                        actual_predicted_diff_signed += predicted_max - actual_max
                    # avoid div by zero
                    den = 0.01 + max(predicted_max, actual_max)

            if model_name not in errors:
                errors[model_name] = []
                signed_errors[model_name] = []
            errors[model_name].append(actual_predicted_diff / den)
            signed_errors[model_name].append(1 if actual_predicted_diff_signed >= 0 else -1)

    return errors, signed_errors, predicted_dict


def get_wrong_predictions(errors_dict, signed_errors_dict, metric, predicted_dict):

    best_predicted = {}
    best_actual = {}
    best_predicted_across_engines = {}
    best_actual_across_engines = {}
    wrong_predictions_dict = {}
    actual_dict = {}

    #print(constants.metric_display_names[metric], predicted_dict)
    for mapping_label, model_errors in errors_dict.items():
        actual_dict[mapping_label] = {}
        for model_name, errors in model_errors.items():
            actual_dict[mapping_label][model_name] = []
            if model_name not in best_predicted:
                best_predicted[model_name] = {}
                best_actual[model_name] = {}
                best_predicted_across_engines[model_name] = []
                best_actual_across_engines[model_name] = []
            
            for num_engines in range(len(errors)):
                predicted = predicted_dict[mapping_label][model_name][num_engines]
                actual_val = predicted * (1 - (errors[num_engines] * signed_errors_dict[mapping_label][model_name][num_engines]))
                if model_name == 'resnet50' and metric == Metrics.LATENCY:
                    print(constants.metric_display_names[metric], mapping_label, model_name, 
                          actual_val, predicted)
                actual_dict[mapping_label][model_name].append(actual_val)
                if num_engines not in best_predicted[model_name]:
                    best_predicted[model_name][num_engines] = [predicted, mapping_label]
                    best_actual[model_name][num_engines] = [actual_val, mapping_label]
                    best_predicted_across_engines[model_name] = [predicted, mapping_label]
                    best_actual_across_engines[model_name] = [actual_val, mapping_label]

                if first_better_than_second(metric, actual_val, best_actual_across_engines[model_name][0]):
                    best_actual_across_engines[model_name] = [actual_val, mapping_label]

                if first_better_than_second(metric, predicted, best_predicted_across_engines[model_name][0]):
                    best_predicted_across_engines[model_name] = [predicted, mapping_label]

                if first_better_than_second(metric, actual_val, best_actual[model_name][num_engines][0]):
                    best_actual[model_name][num_engines] = [actual_val, mapping_label]

                if first_better_than_second(metric, predicted, best_predicted[model_name][num_engines][0]):
                    best_predicted[model_name][num_engines] = [predicted, mapping_label]
            
                #print(constants.metric_display_names[metric], model_name, predicted, actual_val, errors[num_engines])


    for model_name, bests in best_actual.items():
        wrong_predictions_dict[model_name] = []
        for num_engines in range(len(bests)):
            best_act_label = bests[num_engines][1]
            best_predicted_label = best_predicted[model_name][num_engines][1]
            if best_act_label == best_predicted_label or num_engines >= len(predicted_dict[mapping_label][model_name]):
                wrong_predictions_dict[model_name].append(0)
            else:
                predicted_best = best_predicted[model_name][num_engines][0]
                actual_best = bests[num_engines][0]
                relative_diff = abs(predicted_best - actual_best) / max(predicted_best, actual_best)
                wrong_predictions_dict[model_name].append(relative_diff)

        best_act_label = best_actual_across_engines[model_name][1]
        best_predicted_label = best_predicted_across_engines[model_name][1]

        if best_act_label != best_predicted_label:
            predicted_best = best_predicted_across_engines[model_name][0]
            actual_best = best_actual_across_engines[model_name][0]
            relative_diff = abs(predicted_best - actual_best) / actual_best
            wrong_predictions_dict[model_name].append(relative_diff)

    #print(best_predicted_across_engines)
    #print(best_actual_across_engines)

    return wrong_predictions_dict
                

            
            


board_name = 'vcu108'
mapping_list = ['segrr', 'hyb', 'seg']
mpping_labels = {'segrr': SegmentMappingRR.MAPPING_LABEL, 'hyb': HybridMapping.MAPPING_LABEL,
                 'seg': SegmentMapping.MAPPING_LABEL}
min_engines = constants.MIN_ENGINES
max_engines = constants.MAX_ENGINES

metric_dict = {Metrics.BUFFER: constants.metric_display_names[Metrics.BUFFER].lower(),
               Metrics.LATENCY: constants.metric_display_names[Metrics.LATENCY].lower(),
               Metrics.THROUGHPUT: constants.metric_display_names[Metrics.LATENCY].lower()
}

latex_strs = ['&\multicolumn{3}{c|}{\cellcolor{gray!10}Max} & \multicolumn{3}{c|}{\cellcolor{gray!10}Min} \
    & \multicolumn{3}{c|}{\cellcolor{gray!10}Average} & \multicolumn{3}{c|}{\cellcolor{gray!10}Median}\n', '']
latex_errors = {}
metric_id = 0
for metric, metric_dir in metric_dict.items():
    print('>>>>>>>>>>>>>>>>>>>>>', metric)
    latex_errors[metric_id] = {}
    all_erro_vals = {}
    all_errors_dict = {}
    all_signed_errors_dict = {}
    all_predicted_dict = {}
    engine_counts = [*range(min_engines, max_engines + 1)]
    model_name_list = []
    for mapping in mapping_list:
        all_erro_vals[mpping_labels[mapping]] = []
        actual_results = read_actual_results(
            board_name, mapping, metric_dir=metric_dir, min_engines=min_engines, max_engines=max_engines)
        errors, signed_errors, predicted_dict = calc_estimation_errors(
            mapping, actual_results, metric, min_engines, max_engines)
        errors_kes = list(errors.keys())
        errors_vals = list(errors.values())

        all_errors_dict[mpping_labels[mapping]] = copy.deepcopy(errors)
        all_signed_errors_dict[mpping_labels[mapping]] = copy.deepcopy(signed_errors)
        all_predicted_dict[mpping_labels[mapping]] = copy.deepcopy(predicted_dict)

        for i in range(0, len(errors_kes)):
            colors = ['#1f77b4' if e >=
                      0 else '#d62728' for e in errors_vals[i]]
            all_erro_vals[mpping_labels[mapping]].extend(errors_vals[i])
            model_name = errors_kes[i]
            model_name_list.append(model_name)

    wrong_predictions = get_wrong_predictions(all_errors_dict, all_signed_errors_dict, metric, all_predicted_dict)

    #print(constants.metric_display_names[metric], 'wrong_predictions', wrong_predictions)

    x_ticks_dict = {len(model_name_list): engine_counts,
                    len(engine_counts): model_name_list}

    error_summary = {}
    for key, vals in all_erro_vals.items():
        if len(vals) == 0:
            continue
        
        avg_error = sum(vals)
        num_exps = len(vals)
        max_error = max(vals)
        min_error = min(vals)

        avg_error /= num_exps

        avg_acc = 1 - avg_error
        median_acc = 1 - statistics.median(
            vals)
        max_acc = 1 - min_error
        min_acc = 1 - max_error
        error_summary[key] = {'Max': max_acc, 'Min': min_acc,
                              'Average': avg_acc, 'Median': median_acc}

        # latex_errors['min']['headers'].append(key)      
        # latex_errors['max']['headers'].append(key)      
        # latex_errors['avg']['headers'].append(key)      
        # latex_errors['median']['headers'].append(key) 

        if 'max' not in latex_errors[metric_id]:
            latex_errors[metric_id]['max'] = {}
        if 'min' not in latex_errors[metric_id]:
            latex_errors[metric_id]['min'] = {}
        if 'avg' not in latex_errors[metric_id]:
            latex_errors[metric_id]['avg'] = {}
        if 'median' not in latex_errors[metric_id]:
            latex_errors[metric_id]['median'] = {}

        latex_errors[metric_id]['max'][key] = max_acc      
        latex_errors[metric_id]['min'][key] = min_acc      
        latex_errors[metric_id]['avg'][key] = avg_acc      
        latex_errors[metric_id]['median'][key] = median_acc         

    to_dumb = [all_erro_vals, error_summary]

    json_obj = json.dumps(to_dumb)
    with open(constants.FIGURES_DATA_DIR + '/{}/errors/{}_{}_estimation_errors_v2.json'.format(
            board_name, constants.metric_display_names[metric].lower(), board_name), 'w') as f:
        f.write(json_obj)

    to_plot_errors = {}
    for mapping_label in constants.mappings_ordered:
        to_plot_errors[mapping_label] = all_erro_vals[mapping_label]

    latex_strs.append('')
    metric_id += 1
    # plot_estimation_errors(to_plot_errors, '{}_{}_estimation_error'.format(metric_dir, board_name),
    #                        board_name, x_ticks_dict, colors)

#print(latex_errors)
for row, table_row in latex_errors.items():
    for acc_x, mapping_val in table_row.items():
        for mapping, val in mapping_val.items():
            if row == 0:
                latex_strs[row + 1] += '& ' + mapping
            latex_strs[row + 2] += '& ' + str(round(100 * val, 1))
    if row == 0:
        latex_strs[row + 1] += '\\\\ \n'      
    latex_strs[row + 2] += '\\\\ \n'

# for row in latex_strs:
#     print(row)