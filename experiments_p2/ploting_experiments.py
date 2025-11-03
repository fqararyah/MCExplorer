from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
from preformance_record import *
import experiments_p2.experiments as experiments_p2
from plot import plot_results
from plot import plot_utils
from mapping_strategies.mapping_types.mapping_description import *
import experiments_p2.experiments_utils as experiment_utils


def plot_boards_models_metric_vs_all_baselines(metric,
                                               population_size, num_generations,
                                               num_sa_main_iterations,
                                               hm_max_clusters,
                                               saving=False,
                                               plot_ideal = False,
                                               black_box=True,
                                               run_base_anyway=False,
                                               plot_average=True,
                                               sub_plot_params = None,
                                               min_y_lim=0,
                                               _figsize=(8, 1.2),
                                               custom_dir_postfix = '',
                                               file_name = None):
    
    seperate_baseline = True
    include_base=False
    base_dict = experiment_utils.get_bests_in_all_base_mappings_boards_models_and_metrics(
        run_anyway=run_base_anyway,
        custom_dir_postfix = custom_dir_postfix)
    # get_bests_in_all_base_mappings_metrics_boards_models

    ga_dict = experiment_utils.get_genetic_algorithm_experiments_all_metrics_boards_models(num_generations, population_size,
                                                                                           metric_list=[metric],
                                                                                           custom_dir_postfix = custom_dir_postfix,
                                                                                           file_name=file_name)

    sa_dict = experiment_utils.get_simulated_annealing_experiments_all_metrics_boards_models(num_sa_main_iterations,
                                                                                             metric_list=[metric],
                                                                                             custom_dir_postfix = custom_dir_postfix,
                                                                                             file_name=file_name)
    
    hm_dict = experiment_utils.get_hiera_map_algorithm_experiments_all_metrics_boards_models(max_clusters=hm_max_clusters,
                                                                                             metric_list=[metric],
                                                                                             custom_dir_postfix = custom_dir_postfix,
                                                                                             file_name=file_name)

        
    optimized_dicts = {'GA': ga_dict, 'SA': sa_dict, 'HM': hm_dict}
    metric_display_name = constants.metric_display_names[metric]
    
    #get max speedup over best
    optimized_dicts_bb = experiment_utils.get_black_box_dicts(optimized_dicts)
    best_base_dict = experiment_utils.get_best_base_mappings_in_performance_dicts(
        base_dict)
    tmp_norm_dict = experiment_utils.normalize_performance_dicts(
        optimized_dicts_bb, best_base_dict, saving=saving, include_base=False)
    if plot_average:
        experiment_utils.add_board_averages(tmp_norm_dict)
    plot_dict, _, _ = plot_utils.perf_dict_to_plot_dict(
        tmp_norm_dict[metric_display_name])
    bb_norm_list = list(plot_dict[list(plot_dict.keys())[0]])
    max_index = bb_norm_list.index(min(bb_norm_list) if metric == Metrics.ENERGY else max(bb_norm_list))
    #get max speedup over best
    
    if black_box:
        optimized_dicts = experiment_utils.get_black_box_dicts(optimized_dicts)
    
    if plot_ideal and metric in [Metrics.THROUGHPUT, Metrics.LATENCY]:
        ideal_dict = experiment_utils.get_ideal_values_all_metrics_boards_models([
                                                                                 metric])
        optimized_dicts['Ideal'] = ideal_dict

    norm_dict = experiment_utils.normalize_performance_dicts(
        optimized_dicts, base_dict, include_base=include_base, saving=saving, y_threshold = min_y_lim)
    if plot_average:
        experiment_utils.add_board_averages(norm_dict)
    plot_dict, board_names, model_names = plot_utils.perf_dict_to_plot_dict(
        norm_dict[metric_display_name])

    x_ticks_dict = {len(board_names): mapping_general_utils.get_model_display_name_list(model_names),
                    len(model_names): mapping_general_utils.get_board_display_name_list(board_names)}

    ###############################################################
    hatch_dict = {}
    num_groups = len(list(plot_dict.values())[0])
    for mapping_label, vals in plot_dict.items():
        hatch_dict[mapping_label] = []
        for i in range(len(vals)):
            hatch_dict[mapping_label].append(0)

    for group_index in range(num_groups):
        best_label = ''
        best_val = 0
        for mapping_label in constants.mappings_ordered:
            if mapping_label not in plot_dict:
                continue
            current_val = plot_dict[mapping_label][group_index]
            if metric == Metrics.ENERGY:
                if ((best_label == '') or (current_val < best_val)) and (current_val < 1 or not seperate_baseline):
                    best_val = current_val
                    best_label = mapping_label
            else:
                if ((best_label == '') or (current_val > best_val)) and (current_val > 1 or not seperate_baseline):
                    best_val = current_val
                    best_label = mapping_label
        if best_label != '':
            hatch_dict[best_label][group_index] = 1
    print(hatch_dict)
    ################################################################
    plot_results.plot_bar_groups(plot_dict, 'boards_normalized_{}_breakdown'.format(constants.metric_file_names[metric]),
                                 x_title=None,
                                 highlight_group = max_index,
                                 sub_plot_params = sub_plot_params,
                                 hatch_dict = hatch_dict,
                                 _figsize=_figsize,
                                 seperate_baseline_label= constants.mappings_ordered[0] if seperate_baseline else None,
                                 y_title='{}\nReduction'.format(metric_display_name.capitalize()) if saving else
                                 ('Normalized\n{}'.format(metric_display_name))
                                 if metric_display_name.upper() != Metrics.LATENCY.name else 'Speedup',
                                 x_ticks_dict=x_ticks_dict, 
                                 relative_save_path=('bests' + '/{}'.format(custom_dir_postfix)) if custom_dir_postfix != '' else 'bests' ,
                                 abs_save_path=constants.FIGURES_DIR_P2, percentage=saving,
                                 custom_bars_text = bb_norm_list)


def plot_boards_models_metric_vs_best_baseline(metric, population_size, num_generations, num_sa_main_iterations,
                                               hm_max_clusters,
                                               saving=False,
                                               run_base_anyway=False,
                                               plot_ideal=False,
                                               black_box=True,
                                               plot_average=True,
                                               plot_legend = True,
                                               min_y_lim=0,
                                               _figsize=(4, 0.8),
                                               sub_plot_params = None,
                                               custom_dir_postfix = '',
                                               file_name = None):
    base_dict = experiment_utils.get_bests_in_all_base_mappings_boards_models_and_metrics(
        run_anyway=run_base_anyway,
        custom_dir_postfix = custom_dir_postfix)

    ga_dict = experiment_utils.get_genetic_algorithm_experiments_all_metrics_boards_models(num_generations, population_size,
                                                                                           metric_list=[metric],
                                                                                           custom_dir_postfix = custom_dir_postfix,
                                                                                           file_name=file_name)

    sa_dict = experiment_utils.get_simulated_annealing_experiments_all_metrics_boards_models(num_sa_main_iterations,
                                                                                             metric_list=[metric],
                                                                                             custom_dir_postfix = custom_dir_postfix,
                                                                                             file_name=file_name)

    hm_dict = experiment_utils.get_hiera_map_algorithm_experiments_all_metrics_boards_models(max_clusters=hm_max_clusters,
                                                                                             metric_list=[metric],
                                                                                             custom_dir_postfix = custom_dir_postfix)


    if black_box:
        optimized_dicts = {'GA': ga_dict, 'SA': sa_dict, 'HM': hm_dict}
        optimized_dicts = experiment_utils.get_black_box_dicts(optimized_dicts)

        if plot_ideal and metric in [Metrics.THROUGHPUT, Metrics.LATENCY]:
            ideal_dict = experiment_utils.get_ideal_values_all_metrics_boards_models([
                                                                                    metric], optimized_dicts)
            optimized_dicts['Ideal'] = ideal_dict

        best_base_dict = experiment_utils.get_best_base_mappings_in_performance_dicts(
            base_dict)
    else:
        optimized_dicts = {'GA': ga_dict, 'SA': sa_dict, 'HM': hm_dict}
        best_base_dict = experiment_utils.get_best_base_mappings_in_performance_dicts(
            base_dict)
        # for metric_label, metric_dict in hm_dict.items():
        #     best_base_dict[metric_label] = {}
        #     for board_name, board_name_dict in metric_dict.items():
        #         best_base_dict[metric_label][board_name] = {}
        #         for model_name, model_name_dict in board_name_dict.items():
        #             best_base_dict[metric_label][board_name][model_name] = {}
        #             best_base_dict[metric_label][board_name][model_name]['HM'] = hm_dict[metric_label][board_name][model_name]    

    norm_dict = experiment_utils.normalize_performance_dicts(
        optimized_dicts, best_base_dict, saving=saving, include_base=False, y_threshold = min_y_lim)
    
    if plot_average:
        experiment_utils.add_board_averages(norm_dict)
    # print(hm_dict)
    metric_display_name = constants.metric_display_names[metric]
    plot_dict, board_names, model_names = plot_utils.perf_dict_to_plot_dict(
        norm_dict[metric_display_name])

    x_ticks_dict = {len(board_names): mapping_general_utils.get_model_display_name_list(model_names),
                    len(model_names): mapping_general_utils.get_board_display_name_list(board_names)}

    plot_results.plot_bar_groups(plot_dict, 'boards_normalized_{}_best'.format(metric_display_name.lower()) +
                                 ('_bb' if black_box else ''),
                                 sub_plot_params = sub_plot_params, 
                                 x_title=None,
                                 #seperate_baseline_label= 'HM' if not black_box else None,
                                 y_title='{} Reduction'.format(metric_display_name.capitalize()) if saving else
                                 ('Normalized \n {}'.format(metric_display_name))
                                 if metric_display_name.upper() != Metrics.LATENCY.name else 'Speedup',
                                 x_ticks_dict=x_ticks_dict, 
                                 relative_save_path=('bests' + '/{}'.format(custom_dir_postfix)) if \
                                     custom_dir_postfix != '' else 'bests' ,
                                 abs_save_path=constants.FIGURES_DIR_P2,
                                 _figsize = _figsize,
                                 percentage=saving,
                                 min_y_lim=min_y_lim,
                                 plot_legend = plot_legend)
    #seperate_baseline_label='Baseline = 1' if not saving else None,


def plot_pes_models_metric_vs_best_baseline(metric, population_size, num_generations, num_sa_main_iterations,
                                            black_box=True, min_y_lim = 0):
    json_file_path = mapping_general_utils.get_latest_file_path(constants.FIGURES_DATA_DIR_P2 + '/baselines/',
                                                                'bests_in_all_base_mappings_pes_{}.json'.format(
                                                                    metric))
    base_dict = mapping_general_utils.load_json_to_dict(json_file_path)
    json_file_path = mapping_general_utils.get_latest_file_path(
        constants.FIGURES_DATA_DIR_P2 + '/genetic/metrics/',
        ('genetic_algorithm_bests_in_{}_' +
         'population_{}_generations_{}.json').format(metric.lower(), population_size,
                                                     num_generations))
    ga_dict = mapping_general_utils.load_json_to_dict(json_file_path)

    json_file_path = mapping_general_utils.get_latest_file_path(
        constants.FIGURES_DATA_DIR_P2 + '/simulated/metrics/',
        ('simulated_annealing_bests_in_{}_iterations_{}' +
         '.json').format(metric.lower(), num_sa_main_iterations))
    print(json_file_path)
    sa_dict = mapping_general_utils.load_json_to_dict(json_file_path)

    optimized_dicts = {'GA': ga_dict, 'SA': sa_dict}
    
    if black_box:
        optimized_dicts = experiment_utils.get_black_box_dicts(optimized_dicts)

    best_base_dict = experiments_p2.get_best_base_mappings_in_performance_dicts(
        base_dict)
    norm_dict = experiments_p2.normalize_performance_dicts(
        optimized_dicts, best_base_dict)

    plot_dict, board_names, model_names = plot_utils.perf_dict_to_plot_dict(
        norm_dict[metric])

    x_ticks_dict = {len(board_names): mapping_general_utils.get_model_display_name_list(model_names),
                    len(model_names): mapping_general_utils.get_board_display_name_list(board_names)}

    plot_results.plot_bar_groups(plot_dict, 'pes_normalized_{}_best'.format(metric.lower()) +
                                 ('_bb' if black_box else ''),
                                 x_title=None, y_title='Normalized {}'.format(metric),
                                 x_ticks_dict=x_ticks_dict, relative_save_path='bests',
                                 abs_save_path=constants.FIGURES_DIR_P2,
                                 min_y_lim=min_y_lim) #seperate_baseline_label='Baseline = 1', 
