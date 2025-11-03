import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import mapping_utils.mapping_general_utils as mapping_general_utils
import json
from matplotlib.ticker import MultipleLocator
import constants as consts
import plot_utils as plot_utils
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patches as patches
from preformance_record import *
from matplotlib import colormaps
import math
from matplotlib.patches import Patch

path = os.path.dirname(__file__)

def custom_round(x):
    if x >= 1:
        return round(x, 1)
    else:
        return round(x, 2)

def find_optimal_pair(x, y, metric1_lower_is_better=True, metric2_lower_is_better=True):
    import numpy as np

    lx = np.array(x)
    ly = np.array(y)
    n = len(lx)
    #print('============================')


    best_indices = (-1, -1)
    best_coords = ((-1, -1), (-1, -1))

    # Get the top 10% best values
    best_value = np.min(lx) if metric1_lower_is_better else np.max(lx)
    # Accept p1 candidates that are within 10% worse than top_10_best_value
    percents_from_best = 0.1

    while percents_from_best <= 0.9:
        if best_indices[0] != -1:
            break
        degredation_ratio = 0.95
        extended_threshold = best_value * ((1 + percents_from_best) if metric1_lower_is_better else \
                                           (1 - percents_from_best))
        #print(extended_threshold)
        p1_candidates = np.where(lx <= extended_threshold)[0] if metric1_lower_is_better else \
            np.where(lx >= extended_threshold)[0]
        percents_from_best += 0.1
        while degredation_ratio >= 0.89:
            improvement_ratio = 1.05
            if best_indices[0] != -1:
                break
            while improvement_ratio >= max(0.15, 2 * (1- degredation_ratio)):
                best_me_improvement = -np.inf
                if best_indices[0] != -1:
                    break
                for i in p1_candidates:
                    p1_m1 = lx[i]
                    p1_m2 = ly[i]

                    for j in range(n):
                        if j == i:
                            continue

                        p2_m1 = lx[j]
                        p2_m2 = ly[j]

                        # Metric1 degradation: <5% worse
                        if metric1_lower_is_better:
                            m1_degradation_ok = p2_m1 >= p1_m1 and p2_m1 <= p1_m1 * (2 - degredation_ratio)
                        else:
                            m1_degradation_ok = p2_m1 <= p1_m1 and  p2_m1 >= p1_m1 * degredation_ratio

                        # Metric2 improvement: >20% better
                        if metric2_lower_is_better:
                            m2_improvement = p1_m2 - p2_m2
                            m2_improvement_ok = m2_improvement > improvement_ratio * abs(p1_m2)
                        else:
                            m2_improvement = p2_m2 - p1_m2
                            m2_improvement_ok = m2_improvement > improvement_ratio * abs(p1_m2)

                        if p1_m1 < 745 and p1_m1 > 744 and \
                            math.isclose(degredation_ratio, 0.9, rel_tol=1e-5):
                            print(p1_m2, p2_m2, m2_improvement, improvement_ratio * abs(p1_m2), improvement_ratio,
                                  m1_degradation_ok, m2_improvement_ok)
                        # Exclude cases with no real improvement
                        if m1_degradation_ok and m2_improvement_ok and m2_improvement > 0:
                            # Compute ratio
                            #m1_degradation = abs(p2_m1 - p1_m1) / abs(p1_m1) if p1_m1 != 0 else np.inf
                            #ratio = m2_improvement / m1_degradation if m1_degradation != 0 else np.inf

                            if m2_improvement > best_me_improvement:
                                best_me_improvement = m2_improvement
                                best_indices = (i, j)
                                best_coords = ((lx[i], ly[i]), (lx[j], ly[j]))

                improvement_ratio -= 0.05
            degredation_ratio -=0.05

    return best_indices, best_coords

def max_positive_slope_pair(x_list, y_list, x_cooef=1, y_cooef=1, sacrifice_x=False, sacrifice_y=False):
    max_slope = float('-inf')
    best_pair = (-1, -1)
    eps = 10 ** -10
    indices = (-1, -1)
    minimum_meaningful_improvement = 0.15
    sac_x_coeef = -1 if sacrifice_x else 1
    sac_y_coeef = -1 if sacrifice_y else 1
    den_x = 1  # max_x - min_x
    den_y = 1  # max_y - min_y
    n = len(x_list)
    for i in range(n):
        x1, y1 = x_list[i], y_list[i]
        for j in range(n):
            x2, y2 = x_list[j], y_list[j]
            y_diff = (y2 - y1) * y_cooef
            x_diff = (x2 - x1) * x_cooef
            relative_x_diff = x_diff / x1 if not sacrifice_x else 1
            relative_y_diff = y_diff / y1 if not sacrifice_y else 1
            if y1 == y2 or \
                (sacrifice_x and (x2 - x1) * sac_x_coeef * x_cooef < 0) or \
                (sacrifice_y and (y2 - y1) * sac_y_coeef * y_cooef < 0) or \
                    relative_x_diff < minimum_meaningful_improvement or relative_y_diff < minimum_meaningful_improvement:
                continue  # Skip vertical lines (undefined slope)
            y_diff *= (sac_y_coeef / den_y)
            x_diff *= (sac_x_coeef / den_x)
            if x_diff == 0:
                x_diff = eps * (-1 if y_diff < 0 else 1)
            slope = y_diff / x_diff
            slope = (y_cooef * sac_y_coeef / den_y) * (y2 - y1) / (x_diff)
            if slope > 0 and slope > max_slope:
                max_slope = slope
                best_pair = ((x1, y1), (x2, y2))
                indices = (i, j)

    return indices, best_pair, max_slope


def min_positive_slope_pair(x_list, y_list, x_cooef=1, y_cooef=1, sacrifice_x=False, sacrifice_y=False):
    min_slope = float('inf')
    best_pair = (-1, -1)
    indices = (-1, -1)
    min_x = min(x_list)
    min_y = min(y_list)
    max_x = max(x_list)
    max_y = max(y_list)
    sac_x_coeef = -1 if sacrifice_x else 1
    sac_y_coeef = -1 if sacrifice_y else 1
    minimum_meaningful_improvement = 0.15
    eps = 10 ** -10
    den_x = 1  # max_x - min_x
    den_y = 1  # max_y - min_y
    n = len(x_list)
    for i in range(n):
        x1, y1 = x_list[i], y_list[i]
        for j in range(n):
            x2, y2 = x_list[j], y_list[j]
            y_diff = (y2 - y1) * y_cooef
            x_diff = (x2 - x1) * x_cooef
            relative_x_diff = x_diff / x1 if not sacrifice_x else 1
            relative_y_diff = y_diff / y1 if not sacrifice_y else 1
            if y1 == y2 or \
                (sacrifice_x and (x2 - x1) * sac_x_coeef * x_cooef < 0) or \
                (sacrifice_y and (y2 - y1) * sac_y_coeef * y_cooef < 0) or \
                    relative_x_diff < minimum_meaningful_improvement or relative_y_diff < minimum_meaningful_improvement:
                continue  # Skip vertical lines (undefined slope)
            y_diff *= (sac_y_coeef / den_y)
            x_diff *= (sac_x_coeef / den_x)
            if x_diff == 0:
                x_diff = eps * (-1 if y_diff < 0 else 1)
            slope = y_diff / x_diff
            slope = (y_cooef * sac_y_coeef / den_y) * (y2 - y1) / (x_diff)
            if slope > 0 and slope < min_slope:
                min_slope = slope
                best_pair = ((x1, y1), (x2, y2))
                indices = (i, j)

    return indices, best_pair, min_slope


def scatter(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels, plot_name, board_name, model_name,
            custom_engine_counts=None, plot_line=None, figures_path=consts.FIGURES_DIR,
            markers=None, alphas=None, figure_size=None, annotate_max_min=False):

    annotate_worst = False
    annotate_custom = False
    if markers == None:
        markers = ['X', '*', '^', '.']
    save_path = figures_path + '/{}/'.format(board_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += metric + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += model_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dpi_val = 100
    # Show the major grid and style it slightly.
    legend_cols = len(xpoints_grp)
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    else:
        plt.figure(figsize=(3.5, 1))

    legend_y = 1.24
    legend_x = 1
    if custom_engine_counts != None:
        dpi_val = 200

    if alphas is None:
        alphas = [1] * len(xpoints_grp)

    plt.rcParams.update({'font.size': 8})

    plt.grid(which='major', color='#555555', linewidth=1)
    # Show the minor grid as well. Style it in very light gray as a thin,
    plt.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    plt.minorticks_on()
    ax = plt.gca()
    ax.set_axisbelow(True)

    for i in range(len(xpoints_grp)):
        if i == len(xpoints_grp) - 1 and custom_engine_counts != None:
            plt.scatter(xpoints_grp[i], ypoints_grp[i],
                        label=series_labels[i], marker=markers[i], zorder=1, linewidths=0.2, edgecolor='#990000', alpha=.5)
        else:
            plt.scatter(xpoints_grp[i], ypoints_grp[i],
                        label=series_labels[i], marker=markers[i], zorder=i + 2, alpha=alphas[i])
        min_x = min(xpoints_grp[i])
        max_x = max(xpoints_grp[i])
        min_y = min(ypoints_grp[i])
        max_y = max(ypoints_grp[i])
        min_x_index = xpoints_grp[i].index(min_x)
        max_x_index = xpoints_grp[i].index(max_x)
        min_y_index = ypoints_grp[i].index(min_y)
        max_y_index = ypoints_grp[i].index(max_y)

        min_x_index_num_engines = min_x_index + 2
        max_x_index_num_engines = max_x_index + 2
        min_y_index_num_engines = min_y_index + 2
        max_y_index_num_engines = max_y_index + 2

        if annotate_max_min and (i < len(xpoints_grp) - 1 or custom_engine_counts == None or annotate_custom):
            if series_labels[i].lower() == 'custom':
                min_x_index_num_engines = custom_engine_counts[min_x_index]
                max_x_index_num_engines = custom_engine_counts[max_x_index]
                min_y_index_num_engines = custom_engine_counts[min_y_index]
                max_y_index_num_engines = custom_engine_counts[max_y_index]

            y_inc = 0  # ( (len(xpoints_grp) // 2) - i ) / 3
            if plot_name.lower().startswith('thr'):
                ax.annotate(max_x_index_num_engines, (max_x - (max_x_index_num_engines >=
                            10), ypoints_grp[i][max_x_index] + y_inc), weight='bold', zorder=i + 3)
                if annotate_worst and min_x_index != max_x_index:
                    ax.annotate(min_x_index_num_engines, (min_x - (min_x_index_num_engines >=
                                10), ypoints_grp[i][min_x_index] + y_inc), weight='bold', zorder=i + 3)
            else:
                ax.annotate(min_x_index_num_engines, (min_x - (min_x_index_num_engines >=
                            10), ypoints_grp[i][min_x_index] + y_inc), weight='bold', zorder=i + 3)
                if annotate_worst and min_x_index != max_x_index:
                    ax.annotate(max_x_index_num_engines, (max_x - (max_x_index_num_engines >=
                                10), ypoints_grp[i][max_x_index] + y_inc), weight='bold', zorder=i + 3)

            if (min_y_index != min_x_index or not annotate_worst) and min_y_index != max_x_index:
                ax.annotate(min_y_index_num_engines, (xpoints_grp[i][min_y_index] - (min_y_index_num_engines + 2 >= 10),
                                                      min_y + y_inc), weight='bold', zorder=i + 3)
            if annotate_worst and max_y_index != min_x_index and max_y_index != max_x_index:
                ax.annotate(max_y_index_num_engines, (xpoints_grp[i][max_y_index] - (max_y_index_num_engines + 2 >= 10),
                                                      max_y + y_inc), weight='bold', zorder=i + 3)

    if plot_line is not None:
        print(save_path + '{}.png'.format(plot_name.lower()))
        line_label = list(plot_line.keys())[0]
        line_y = list(plot_line.values())[0]
        plt.axhline(y=line_y, label=line_label, linestyle='-', color='red')

    plt.xlabel(axis_labels[0], labelpad=0.2)
    plt.ylabel(axis_labels[1], labelpad=0)

    plt.legend(loc='upper right', bbox_to_anchor=(legend_x, legend_y),
               ncol=legend_cols, handletextpad=0.1, handlelength=0.8, columnspacing=0.4,
               frameon=False, borderpad=0)
    # plt.title(board_name + '_' + model_name)
    plt.savefig(save_path + '{}.png'.format(plot_name.lower()), format='png',
                bbox_inches='tight', dpi=dpi_val)
    plt.savefig(save_path + '{}.pdf'.format(plot_name.lower()), format='pdf',
                bbox_inches='tight', dpi=dpi_val)
    dump_dict = {'x': xpoints_grp, 'y': ypoints_grp}
    json_obj = json.dumps(dump_dict)
    with open(save_path + '{}.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    plt.clf()
    plt.close()


def scale_x_and_y(xpoints_grp, ypoints_grp, scale_x, scale_y):
    for i in range(len(xpoints_grp)):
        xpoints_grp[i] *= scale_x
        ypoints_grp[i] *= scale_y


def scatter_one_series(xpoints, ypoints, axis_labels, plot_name, save_path, figure_size=None, marker='o', alpha=0.6,
                       set_xlimit_to_data=False, gradient=False, higher_is_better=True):
    plt.rcParams.update({'font.size': 8})
    fig = plt.figure(figsize=figure_size)
    ax = plt.subplot(111)  # whole path
    fig.tight_layout()
    ax.grid(which='major', color='#555555', linewidth=1)
    ax.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)

    min_y = min(ypoints)
    max_y = max(ypoints)

    gradient_colors = [(y - min_y) / (max_y - min_y) for y in ypoints]
    if not higher_is_better:
        gradient_colors = [1-color for color in gradient_colors]
    if gradient:
        ax.scatter(xpoints, ypoints, marker=marker, alpha=alpha,
                   c=gradient_colors, cmap='Blues', edgecolor='black')
    else:
        ax.scatter(xpoints, ypoints, marker=marker, alpha=alpha)

    if set_xlimit_to_data:
        ax.set_xlim(min(xpoints), max(xpoints))

    ax.set_xlabel(axis_labels[0], labelpad=0.2)
    ax.set_ylabel(axis_labels[1], labelpad=0)

    plt.savefig(save_path + '{}.png'.format(plot_name.lower()), format='png',
                bbox_inches='tight')
    plt.savefig(save_path + '{}.pdf'.format(plot_name.lower()), format='pdf',
                bbox_inches='tight')
    dump_dict = {'x': xpoints, 'y': ypoints}
    json_obj = json.dumps(dump_dict)
    with open(save_path + '{}.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    plt.clf()
    plt.close()


def scatter_with_zoom(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels,
                      plot_name, board_name, model_name,
                      figures_path=consts.FIGURES_DIR, markers=[
                          'X', '*', '^', '.'],
                      series_to_zoom_index=-1,
                      alphas=None, figure_size=None, legend_specs={'location': 'upper right',
                                                                   'ncol': 1},
                      scale_x=1,
                      scale_y=1,
                      gradient=False):

    if figure_size is None:
        figure_size = (3.5, 1)

    cmaps = list(colormaps)
    fig = plt.figure(figsize=figure_size)
    ax = plt.subplot(111)  # whole path
    fig.tight_layout()
    ax.grid(which='major', color='#555555', linewidth=1)
    ax.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': 8})

    trade_off_points = {}
    duplicate_tradeoff_points = {}
    tradeoff_point_rank = 1

    markersize = 6  # 12 is the default

    higher_is_better_x = 'thr' in axis_labels[0].lower()
    higher_is_better_y = 'thr' in axis_labels[1].lower()

    if alphas is None:
        alphas = [1] * len(xpoints_grp)

    for i in range(len(xpoints_grp)):
        scale_x_and_y(xpoints_grp[i], ypoints_grp[i], scale_x, scale_y)

    # Define triangle in axes coordinates (0 = left/bottom, 1 = right/top)
    corner_1 = [1, 1]
    corner_2 = [0.8, 1]
    corner_3 = [1, 0.65]
    if higher_is_better_x and not higher_is_better_y:
        corner_1 = [1, 0]
        corner_2 = [0.8, 0]
        corner_3 = [1, 0.35]
    elif higher_is_better_y:
        corner_1 = [0, 1]
        corner_2 = [0, 0.65]
        corner_3 = [0.2, 1]
    else:
        corner_1 = [0, 0]
        corner_2 = [0, 0.35]
        corner_3 = [0.2, 0]

    triangle = patches.Polygon(
        # Triangle points: bottom-left, right, top
        [corner_1, corner_2, corner_3],
        closed=True,
        transform=ax.transAxes,         # Interpret in axes coordinates!
        color='purple',
        alpha=0.1,
        clip_on=False
    )

    for i in range(len(xpoints_grp)):
        min_x = min(xpoints_grp[i])
        max_x = max(xpoints_grp[i])
        min_y = min(ypoints_grp[i])
        max_y = max(ypoints_grp[i])
        gradient_colors_y = [
            (y - min_y) / (0.00000000001 + max_y - min_y) for y in ypoints_grp[i]]
        gradient_colors_x = [
            (x - min_x) / (0.00000000001 + max_x - min_x) for x in xpoints_grp[i]]
        if not higher_is_better_x:
            gradient_colors_x = [1-color for color in gradient_colors_x]
        if not higher_is_better_y:
            gradient_colors_y = [1-color for color in gradient_colors_y]
        gradient_colors = [(gradient_colors_x[i] + gradient_colors_x[i]
                            ) / 2 for i in range(len(gradient_colors_x))]
        if gradient:
            ax.scatter(xpoints_grp[i], ypoints_grp[i],
                       cmap=cmaps[i],
                       c=gradient_colors,
                       label=series_labels[i], marker=markers[i], alpha=alphas[i])
        else:
            ax.scatter(xpoints_grp[i], ypoints_grp[i],
                       label=series_labels[i], marker=markers[i], alpha=alphas[i])

    ax.legend(loc=legend_specs['location'], ncol=legend_specs['ncol'], handlelength=0.5,
              columnspacing=0.2, borderpad=0.1, labelspacing=0.1, handletextpad=0.1)

    series_to_zoom_x_list = xpoints_grp[series_to_zoom_index]
    series_to_zoom_y_list = ypoints_grp[series_to_zoom_index]
    points_in_inche = 72  # Convert 1 pt to inches
    x_span = (ax.get_xlim()[1] - ax.get_xlim()[0])
    y_span = (ax.get_ylim()[1] - ax.get_ylim()[0])
    point_in_fig_coords_x = x_span / (points_in_inche * fig.get_figwidth())
    point_in_fig_coords_y = y_span / (points_in_inche * fig.get_figheight())
    zoom_min_x = min(series_to_zoom_x_list)
    zoom_max_x = max(series_to_zoom_x_list)
    zoom_range_x = zoom_max_x - zoom_min_x
    zoom_min_y = min(series_to_zoom_y_list)
    zoom_max_y = max(series_to_zoom_y_list)
    zoom_range_y = zoom_max_y - zoom_min_y

    zoom_min_y_rect = zoom_min_y - 0.25 * \
        markersize * point_in_fig_coords_y
    zoom_max_y_rect = zoom_max_y + 0.5 * \
        markersize * point_in_fig_coords_y

    zoom_min_x_rect = zoom_min_x - 0.25 * \
        markersize * point_in_fig_coords_x
    zoom_max_x_rect = zoom_max_x + 0.5 * \
        markersize * point_in_fig_coords_x

    axins_width = fig.get_figwidth() * 0.8
    axins_height = fig.get_figheight() / 2

    axins = inset_axes(ax, axins_width, axins_height, loc='upper left',
                       axes_kwargs={"facecolor": "white"}, bbox_to_anchor=(0, 1.8), bbox_transform=ax.transAxes)

    x_cooef = 1 if higher_is_better_x else -1
    y_cooef = 1 if higher_is_better_y else -1

    # (point1_index, point2_index), (point1, point2), _ = max_positive_slope_pair(x_list=xpoints_grp[-1],
    #                                                              y_list=ypoints_grp[-1],
    #                                                              x_cooef=x_cooef,
    #                                                              y_cooef=y_cooef,
    #                                                              sacrifice_x= True)

    (point1_index, point2_index), (point1, point2) = find_optimal_pair(xpoints_grp[-1], ypoints_grp[-1],
                                                                       not higher_is_better_x,
                                                                       not higher_is_better_y)
    if point1_index != -1:
        for key, val in trade_off_points.items():
            if val[0] == point1[0] and val[1] == point1[1]:
                duplicate_tradeoff_points[tradeoff_point_rank] = 1
            if val[0] == point2[0] and val[1] == point2[1]:
                duplicate_tradeoff_points[tradeoff_point_rank + 1] = 1

        trade_off_points[tradeoff_point_rank] = point1
        tradeoff_point_rank += 1
        trade_off_points[tradeoff_point_rank] = point2
        tradeoff_point_rank += 1

    (point1_index, point2_index), (point1, point2) = find_optimal_pair(ypoints_grp[-1], xpoints_grp[-1],
                                                                       not higher_is_better_y,
                                                                       not higher_is_better_x)

    if point1_index != -1:
        #print((point1_index, point2_index), (point1, point2))
        for key, val in trade_off_points.items():
            if val[0] == point1[1] and val[1] == point1[0]:
                duplicate_tradeoff_points[tradeoff_point_rank] = 1
            if val[0] == point2[1] and val[1] == point2[0]:
                duplicate_tradeoff_points[tradeoff_point_rank + 1] = 1

        trade_off_points[tradeoff_point_rank] = (point1[1], point1[0])
        tradeoff_point_rank += 1
        trade_off_points[tradeoff_point_rank] = (point2[1], point2[0])
        tradeoff_point_rank += 1

    for key, val in trade_off_points.items():
        annotation_key = key
        if key in duplicate_tradeoff_points:
            annotation_key = '*'

        axins.annotate(
            annotation_key, (val[0], val[1]), weight='bold', zorder=i + 3)

    # axins.patch.set_alpha(0.5)
    for i in range(len(xpoints_grp)):
        # plot the same data on both axes
        axins.scatter(xpoints_grp[i], ypoints_grp[i],
                      label=series_labels[i], marker=markers[i], alpha=alphas[i])

    axins.set_xlim(zoom_min_x - zoom_range_x / 20,
                   zoom_max_x + zoom_range_x / 10)
    axins.set_ylim(zoom_min_y - zoom_range_y / 20,
                   zoom_max_y + zoom_range_y / 10)

    axins.grid(which='major', color='#555555', linewidth=1)
    axins.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    axins.minorticks_on()
    axins.set_axisbelow(True)
    axins.xaxis.tick_top()
    axins.yaxis.tick_right()

    rect_min_x = zoom_min_x_rect
    rect_width = (zoom_range_x) + \
        0.5 * markersize * point_in_fig_coords_x
    rect_min_y = zoom_min_y_rect
    rect_height = zoom_max_y_rect - zoom_min_y_rect

    ax.scatter(rect_min_x, rect_min_y, color='none')
    ax.scatter(rect_min_x + rect_width, rect_min_y, color='none')

    conn = patches.ConnectionPatch(
        xyA=(rect_min_x, rect_min_y), coordsA='data', axesA=ax,
        xyB=(axins.get_xlim()[0], axins.get_ylim()[0]), coordsB='data', axesB=axins,
        color='black',
    )
    ax.add_artist(conn)
    conn.set_in_layout(False)  # remove from layout calculations

    conn2 = patches.ConnectionPatch(
        xyA=(rect_min_x + rect_width, rect_min_y), coordsA='data', axesA=ax,
        xyB=(axins.get_xlim()[1], axins.get_ylim()[0]), coordsB='data', axesB=axins,
        color='black',
    )
    ax.add_artist(conn2)
    conn2.set_in_layout(False)  # remove from layout calculations

    ax.add_patch(triangle)

    rect = patches.Rectangle((rect_min_x, rect_min_y), rect_width,
                             rect_height, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    ax.set_xlabel(axis_labels[0], labelpad=0.2)
    ax.set_ylabel(axis_labels[1], labelpad=0)

    save_path = mapping_general_utils.prepare_save_path_from_list(
        figures_path, [board_name, metric, model_name])

    plt.savefig(save_path + '{}_zoom.png'.format(plot_name.lower()), format='png',
                bbox_inches='tight')
    plt.savefig(save_path + '{}_zoom.pdf'.format(plot_name.lower()), format='pdf',
                bbox_inches='tight')
    dump_dict = {'x': xpoints_grp, 'y': ypoints_grp}
    json_obj = json.dumps(dump_dict)
    with open(save_path + '{}.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    json_obj = json.dumps(trade_off_points)
    with open(save_path + '{}_trade_off_points.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    plt.clf()
    plt.close()


def scatter_with_breaks(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels,
                        plot_name, board_name, model_name,
                        figures_path=consts.FIGURES_DIR, markers=None,
                        alphas=None, figure_size=None, legend_specs={'location': 'lower left',
                                                                     'ncol': 1}):

    if figure_size is None:
        figure_size = (3.5, 1)

    fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w',
                                  figsize=figure_size, gridspec_kw={'width_ratios': [1, 1]})
    fig.tight_layout()

    ax.grid(which='major', color='#555555', linewidth=1)
    ax.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax2.grid(which='major', color='#555555', linewidth=1)
    ax2.grid(which='minor', color='#AAAAAA', linewidth=0.5)
    ax2.minorticks_on()
    ax2.set_axisbelow(True)
    plt.rcParams.update({'font.size': 8})

    plt.subplots_adjust(wspace=0.05)

    if alphas is None:
        alphas = [1] * len(xpoints_grp)

    if markers == None:
        markers = ['X', '*', '^', '.']

    x_break_point1, x_break_point2, range_min = plot_utils.decide_break_points_by_heursitic_results(
        xpoints_grp)

    print('>>>', x_break_point1, x_break_point2)

    for i in range(len(xpoints_grp)):
        # plot the same data on both axes
        ax.scatter(xpoints_grp[i], ypoints_grp[i],
                   label=series_labels[i], marker=markers[i], alpha=alphas[i])
        if i != len(xpoints_grp) - 1:
            ax2.scatter(xpoints_grp[i], ypoints_grp[i],
                        label=series_labels[i], marker=markers[i], alpha=alphas[i])

    ax.set_xlim(range_min - (x_break_point1 - range_min) / 40, x_break_point1)
    ax2.set_xlim(x_break_point2, ax2.get_xlim()[1])

    ax.legend(loc=legend_specs['location'], ncol=legend_specs['ncol'], handlelength=0.8,
              columnspacing=0.4, borderpad=0.1, labelspacing=0.1)

    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False, right=False)
    ax2.tick_params(which='minor', left=False)

    d = .1  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1, 1), (-d, +d), **kwargs)
    ax.plot((1, 1), (1-d, 1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((0, 0), (1-d, 1+d), **kwargs)
    ax2.plot((0, 0), (-d, +d), **kwargs)

    ax.set_xlabel(axis_labels[0], labelpad=0.2)
    ax.set_ylabel(axis_labels[1], labelpad=0)

    save_path = mapping_general_utils.prepare_save_path_from_list(
        figures_path, [board_name, metric, model_name])

    plt.savefig(save_path + '{}_break.png'.format(plot_name.lower()), format='png',
                bbox_inches='tight')
    plt.savefig(save_path + '{}_break.pdf'.format(plot_name.lower()), format='pdf',
                bbox_inches='tight')
    dump_dict = {'x': xpoints_grp, 'y': ypoints_grp}
    json_obj = json.dumps(dump_dict)
    with open(save_path + '{}.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    plt.clf()
    plt.close()


def scatter_with_breaks_xy(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels,
                           plot_name, board_name, model_name,
                           figures_path=consts.FIGURES_DIR, markers=None,
                           alphas=None, figure_size=None,):
    if figure_size is None:
        figure_size = (10, 10)

    # fig = plt.figure()
    # gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    # (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col',
                                               sharey='row', facecolor='w',
                                               figsize=figure_size,
                                               gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]})

    if alphas is None:
        alphas = [1] * len(xpoints_grp)

    if markers == None:
        markers = ['X', '*', '^', '.']

    x_break_point1, x_break_point2, range_min, range_max = plot_utils.decide_break_points(
        xpoints_grp, ypoints_grp, 0.35, 0.1)
    y_break_point1, y_break_point2, y_range_min, range_max_y = plot_utils.decide_break_points(
        ypoints_grp, xpoints_grp, 0.1, 0.35)
    # print('>>>', x_break_point1, x_break_point2)
    # print('>>>', y_break_point1, y_break_point2)

    for i in range(len(xpoints_grp)):
        # plot the same data on both axes
        ax1.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i])
        ax2.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i])
        ax3.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i])
        ax4.scatter(xpoints_grp[i], ypoints_grp[i],
                    label=series_labels[i], marker=markers[i])

    margin = 0  # (x_break_point1 - range_min) / 10
    starting_point_left = range_min - margin
    ax1.set_xlim(starting_point_left, x_break_point1 + margin)
    ax3.set_xlim(starting_point_left, x_break_point1 + margin)

    margin = 0  # (range_max - x_break_point2) / 10
    ax2.set_xlim(x_break_point2 - margin, ax2.get_xlim()[1])
    ax4.set_xlim(x_break_point2 - margin, ax2.get_xlim()[1])

    margin = 0  # (y_break_point1 - y_range_min) / 10
    starting_point_bottom = y_range_min - margin
    ax3.set_ylim(starting_point_bottom, y_break_point1 + margin)
    ax4.set_ylim(starting_point_bottom, y_break_point1 + margin)

    margin = 0  # (range_max_y - y_break_point2) / 10
    ax1.set_ylim(y_break_point2 - margin, ax2.get_ylim()[1])
    ax2.set_ylim(y_break_point2 - margin, ax2.get_ylim()[1])

    # hide the spines
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(left=False, right=False)

    ax3.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    ax4.tick_params(left=False, right=False)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax1.tick_params(bottom=False)
    ax2.tick_params(bottom=False)

    ax3.spines['top'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax3.tick_params(top=False)
    ax4.tick_params(top=False)

    d = .03  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
    ax3.plot((1-d, 1+d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
    ax4.plot((-d, +d), (-d, +d), **kwargs)

    plt.xlabel(axis_labels[0], labelpad=0.2)
    plt.ylabel(axis_labels[1], labelpad=0)

    save_path = mapping_general_utils.prepare_save_path_from_list(
        figures_path, [board_name, metric, model_name])

    plt.savefig(save_path + '{}_break_xy.png'.format(plot_name.lower()), format='png',
                bbox_inches='tight')
    plt.savefig(save_path + '{}_break_xy.pdf'.format(plot_name.lower()), format='pdf',
                bbox_inches='tight')
    dump_dict = {'x': xpoints_grp, 'y': ypoints_grp}
    json_obj = json.dumps(dump_dict)
    with open(save_path + '{}.json'.format(plot_name.lower()), 'w') as f:
        f.write(json_obj)
    plt.clf()
    plt.close()


def plot_bars_as_interleaved_bar_groups(num_groups, plot_dict, colors=[], x_ticks_dict=None,
                                        rotation='horizontal', secondary_ticks_location=0.25,
                                        y_title=None,
                                        legend_y=1, legend_x=0, bar_linewidth=2, breakdown=False,
                                        horizontal_bars=False,
                                        seperate_baseline_label=None,
                                        make_space_between_axis_and_bars=False,
                                        draw_norm_line=False,
                                        percentage=False,
                                        plot_average=False,
                                        legend_n_col = -1,
                                        min_y_lim=0,
                                        hatch_dict = None,
                                        sub_plot_params = None,
                                        plot_legend = True,
                                        highlight_group = -1,
                                        custom_bars_text = None,
                                        text_rotation = 'horizontal',
                                        show_only_best_label = False):
    x = []
    num_bars_in_group = len(plot_dict)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    is_seperate_baseline = 0
    handles = []
    if seperate_baseline_label is not None:
        is_seperate_baseline = 1
        handles.append(Patch(facecolor=default_colors[0], label=seperate_baseline_label, edgecolor='black', linewidth=bar_linewidth))
        plt.axhline(y=1 - min_y_lim, label=seperate_baseline_label, linestyle='-',
                    color=default_colors[0], linewidth=2, zorder=10)
    
    if legend_n_col == -1:
        legend_n_col = num_bars_in_group + is_seperate_baseline + plot_average

    print(num_bars_in_group)
    if breakdown:
        num_bars_in_group = 1

    column_width = 1 / (num_bars_in_group + 1)
    # if column_width == 0.5:
    #     column_width = 0.8

    if x_ticks_dict is not None:
        xticks_vals = list(x_ticks_dict.values())
        xticks_repititions = list(x_ticks_dict.keys())
        num_first_level_ticks = len(xticks_vals[0])
        ticks_0 = xticks_vals[0] * xticks_repititions[0]

    for j in range(num_bars_in_group):
        # adding space between sets of groups if there are two levels on x-axis
        x.append([])
        first_level_offsex = 0
        for i in range(num_groups):
            if horizontal_bars and i % num_first_level_ticks == 0:
                first_level_offsex += column_width
            x[j].append(i + column_width * j + first_level_offsex)

    bottom = np.zeros(num_groups)

    ylim = 0
    max_y = 0
    min_y = 1000000
    j = 0
    color_index = 0
    align = 'edge'
    bests = []
    there_is_ideal = 0
    if breakdown or num_bars_in_group == 1:
        align = 'center'
    for series, values in plot_dict.items():
        current_color = default_colors[is_seperate_baseline + color_index]
        handles.append(Patch(facecolor=current_color, label=series, edgecolor='black', linewidth=bar_linewidth))
        if series.lower() in ['ideal', 'perfect']:
            current_color = '#ffffff50'
            there_is_ideal = 1
            handles[-1] = Patch(facecolor=current_color, label=series, edgecolor='black', linewidth=bar_linewidth)
        else:
            max_y = max(max_y, max(values))
            min_y = min(min_y, min(values))
            for i in range(len(values)):
                if i == len(bests):
                    bests.append(values[i])
                if values[i] > bests[i]:
                    bests[i] = values[i]

            color_index += 1
        ylim = max(ylim, max(values))
        if horizontal_bars:
            bars = plt.barh(x[j], values, column_width,
                     label=series, edgecolor='black', zorder=3, linewidth=bar_linewidth, left=bottom, align=align)
        else:
            bars = plt.bar(x[j], values, column_width,
                        label=series, edgecolor='black', color=current_color, zorder=3,
                        linewidth=bar_linewidth, bottom=bottom, align=align)
        
        if j == num_bars_in_group - 1:
            for grp_index in range(num_groups):
                if grp_index == highlight_group:
                    fontweight='bold'
                    label_facecolor=default_colors[num_bars_in_group - there_is_ideal]
                    label_color = 'white'
                    label_linewidth = 1
                else:
                    if show_only_best_label:
                        continue
                    fontweight='normal'
                    label_facecolor='lightgrey'
                    label_color = 'black'
                    label_linewidth = 0
                if custom_bars_text is not None:
                    plt.text(
                        x[0][grp_index] + column_width * num_bars_in_group / 2,
                        ylim - 0.1,
                        str(custom_round(custom_bars_text[grp_index])),
                        ha='center',
                        va='top',
                        rotation = text_rotation,
                        color = label_color,
                        fontweight=fontweight,
                        bbox=dict(
                        edgecolor='black', 
                        linewidth = label_linewidth,
                        facecolor=label_facecolor,   # Background color
                        #alpha=0.4,           # Transparency level (0.0–1.0)
                        pad=0,
                    )
                    )
                #if grp_index == highlight_group:
                    # cbars = plt.bar(x[0][grp_index], ylim, column_width * (num_bars_in_group - there_is_ideal), align=align,
                    #     color='white', alpha=0.3, zorder=-1, edgecolor='blue', linewidth=1)
                    # for bar in cbars:
                    #     bar.set_hatch('xx')

        if hatch_dict is not None:
            grp_index = 0
            for bar, hatch in zip(bars, hatch_dict[series]):
                if hatch == 1:
                    bar.set_hatch('////')
        if breakdown:
            bottom += values
        else:
            j += 1

    if make_space_between_axis_and_bars:
        if breakdown:
            plt.xlim([- 0.5, len(x[0]) - 0.5])
        else:
            plt.xlim([- 0.1, len(x[0]) - 0.2])
    
    if not horizontal_bars:
        if percentage:
            plt.ylim([min_y_lim, ylim + 0.01])
            plt.yticks(np.arange(min_y_lim, ylim + 0.01, ylim / 2))

        if percentage:
            plt.gca().yaxis.set_minor_locator(MultipleLocator(ylim / 4))
        elif ylim >= 2:
            plt.gca().yaxis.set_minor_locator(MultipleLocator(0.5))
        else:
            plt.gca().yaxis.set_minor_locator(MultipleLocator(0.25))

    if min_y_lim != 0:
        yticks = plt.yticks()[0]  # Get current tick locations
        plt.yticks(yticks, [f'{min_y_lim + ytick:.2f}'.rstrip('0').rstrip('.') for ytick in yticks])
        plt.ylim(min_y, max_y)
    if min_y > 1:
        plt.ylim(1, max_y)
        
    avg = sum(bests) / len(bests)
    if plot_average:
        plt.axhline(y=avg, label='Average = ' + str(round(avg, 2)), linestyle='-',
                    color=default_colors[is_seperate_baseline + color_index + 1], linewidth=3)

    # if len(plot_dict) > 1:

    if (sub_plot_params is None or sub_plot_params[-1] == 1) and \
        plot_legend:
        if horizontal_bars:
            plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(
                legend_x, legend_y), frameon=False, borderpad=0, ncol=legend_n_col)
        else:
            plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(legend_x, legend_y),
                    ncol=legend_n_col, frameon=False, borderpad=0)

    if sub_plot_params is None or (sub_plot_params[0] == sub_plot_params[-1]) or sub_plot_params[0] == 1:
        if x_ticks_dict is not None:
            if horizontal_bars:
                plt.yticks(x[len(x)//2], ticks_0)
            else:
                plt.xticks(x[len(x)//2], ticks_0, rotation=rotation)
    else:
        plt.tick_params(labelbottom=False)

    if x_ticks_dict is not None and len(xticks_vals) > 1:
        ticks_1 = xticks_vals[1]
        if len(ticks_1) > 1:
            secondary_x = []
            step = num_groups / len(ticks_1)
            offset = step / 2
            for i in range(len(ticks_1)):
                secondary_x.append(offset)
                offset += step
                if horizontal_bars:
                    offset += column_width
                else:
                    if i > 0:
                        plt.axvline(x=i * step - column_width,
                                    color='black', linewidth=1)

            if horizontal_bars:
                sec = plt.gca().secondary_yaxis(secondary_ticks_location)
                sec.set_yticks(secondary_x, ticks_1, rotation='vertical')
            else:
                sec = plt.gca().secondary_xaxis(secondary_ticks_location)
                sec.set_xticks(secondary_x, ticks_1)


def separate_bars(xpoints_grp, plot_dict, colors=[]):
    x = np.arange(len(xpoints_grp[0]))  # the label locations
    bar_width = 0.8
    width = len(x)  # the width of the bars
    multiplier = 0

    xticks_x = []
    xticks_y = []
    seperator = 2

    for i in range(len(xpoints_grp)):
        for j in range(0, len(x), 2):
            xticks_x.append(i * (len(x) + seperator) + j)
            xticks_y.append(2 + j)

    for series, values in plot_dict.items():
        offset = (width + seperator) * multiplier
        if len(colors) > 1:
            rects = plt.bar(x + offset, values, bar_width,
                            label=series, color=colors, edgecolor='black')
        else:
            rects = plt.bar(x + offset, values, bar_width,
                            label=series, edgecolor='black')

        # plt.broken_barh([(0, 15)], (2, 4))
        multiplier += 1

    plt.xticks(xticks_x, xticks_y)


def prepare_plot_dict(xpoints_grp, ypoints_grp, series_labels, normalize):
    plot_dict = {}
    for i in range(len(xpoints_grp)):
        if normalize:
            normalization_base = ypoints_grp[i][0]
            for j in range(len(ypoints_grp[i])):
                ypoints_grp[i][j] /= normalization_base
        plot_dict[series_labels[i]] = ypoints_grp[i]

    return plot_dict

# x_ticks_dict: for hierarical or multiple ticks. The format: {num_bar_groups_per_tick: [ticks]}


def plot_bar_groups(data_dict, plot_file_name,
                    sub_plot_params = None,
                    x_title=None, y_title=None, x_ticks_dict=None, relative_save_path=None, interleaved_bars=True,
                    breakdown=False, horizontal_bars=False, percentage=False,
                    seperate_baseline_label=None, abs_save_path=consts.FIGURES_DATA_DIR,
                    plot_average=False,
                    min_y_lim=0,
                    _figsize = None,
                    hatch_dict = None,
                    highlight_group = -1,
                    plot_legend = True,
                    custom_bars_text = None):

    save_path = mapping_general_utils.prepare_save_path(
        abs_save_path, relative_save_path)
    num_groups = len(list(data_dict.values())[0])
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['xtick.major.pad'] = 0
    plt.rcParams['ytick.major.pad'] = 0
    if sub_plot_params is not None:
        plt.subplot(*sub_plot_params)
    else:
        ax = plt.axes([0, 0, 1, 1])
    rotation = 'horizontal'
    secondary_ticks_location = -0.25
    dpi_val = 100
    legend_x = 0.5
    legend_y = 0
    draw_norm_line = True
    make_space_between_axis_and_bars = False
    bar_linewidth = 1
    legend_n_col = -1
    text_rotation = 'horizontal'
    show_only_best_label = False
    if sub_plot_params is None:
        if horizontal_bars:
            secondary_ticks_location = -0.42
            plt.figure(figsize=(3.2, 0.5))
            legend_x = 0.2
            legend_y = 1.6
        elif num_groups == 2:
            secondary_ticks_location = -0.5
            plt.figure(figsize=(3.45, 1))
            legend_y = 1.25
        elif num_groups == 8:
            rotation = 20
            secondary_ticks_location = -0.5
            plt.figure(figsize=(3.45, 0.8))
            legend_y = 1.15
        elif num_groups == 20:
            rotation = 'vertical'
            secondary_ticks_location = -0.6
            plt.figure(figsize=(6.45, 1.2))
            legend_y = 1.17
        elif num_groups == 11:
            draw_norm_line = False
            secondary_ticks_location = -0.6
            plt.figure(figsize=(3.45, 1))
            legend_y = 1
        elif num_groups == 40:
            secondary_ticks_location = -0.4
            plt.figure(figsize=(8, 0.6))
            legend_y = 1.4
        elif num_groups == 27:
            rotation = 'vertical'
            draw_norm_line = False
            secondary_ticks_location = -0.4
            plt.figure(figsize=(3.45, 0.6))
            legend_y = 1.45
        elif num_groups == 7:
            draw_norm_line = False
            secondary_ticks_location = -0.4
            plt.figure(figsize=(3.45, 0.6))
            legend_y = 1.45
        elif num_groups == 50:
            rotation = 'vertical'
            draw_norm_line = False
            secondary_ticks_location = -0.6
            plt.figure(figsize=(10, 0.5))
            legend_y = 1.55
        if num_groups == 2 and breakdown and not horizontal_bars:
            plt.figure(figsize=(1.6, 1))
            legend_y = 1
            legend_x = 0.3
            legend_y = 1.3
            make_space_between_axis_and_bars = True
        if num_groups == 4 and len(data_dict) == 2 and not horizontal_bars:
            plt.figure(figsize=(1.6, 1))
            legend_x = 0.4
            legend_y = 1.29
            make_space_between_axis_and_bars = True

        if _figsize is not None:
            if _figsize[0] == 8:
                plt.figure(figsize=_figsize)
                legend_y = 1.17
                rotation = 'vertical'
                secondary_ticks_location = -0.6
            elif _figsize[0] == 4:
                plt.figure(figsize=_figsize)
                legend_y = 1.25
                rotation = 'vertical'
                secondary_ticks_location = -0.82
            elif _figsize[0] == 2.6:
                show_only_best_label = True
                #text_rotation = 'vertical'
                legend_n_col = 2
                plt.figure(figsize=_figsize)
                legend_y = 1.3
                rotation = 'vertical'
                secondary_ticks_location = -0.55
    else:
        legend_y = 1.2
        rotation =45
        secondary_ticks_location = -0.5
        if sub_plot_params[0] == 1:
            rotation = 'vertical'
            secondary_ticks_location = -0.65
            
        plt.margins(0)
    # plt.gca().tick_params(axis='y', which='minor', length=2)

    # plt.minorticks_on()
    if horizontal_bars:
        plt.grid(axis='x', color='#555555', which='major')
        if sub_plot_params is None or sub_plot_params[0] != 1:
            plt.grid(axis='x', color='#555555', which='minor',
                    linewidth=0.5, linestyle='dashed')
    else:
        plt.grid(axis='y', color='#555555', which='major')
        if sub_plot_params is None or sub_plot_params[0] != 1:
            plt.grid(axis='y', color='#555555', which='minor',
                 linewidth=0.5, linestyle='dashed')
    plt.ylabel(y_title, labelpad=0.4)
    plt.xlabel(x_title, labelpad=0.4)
    plt.autoscale(enable=True, axis='x', tight=True)

    if percentage:
        if horizontal_bars:
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        else:
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    if interleaved_bars:
        plot_bars_as_interleaved_bar_groups(num_groups=num_groups, plot_dict=data_dict, x_ticks_dict=x_ticks_dict,
                                            rotation=rotation, secondary_ticks_location=secondary_ticks_location,
                                            y_title=y_title,
                                            legend_y=legend_y, legend_x=legend_x, bar_linewidth=bar_linewidth,
                                            breakdown=breakdown, horizontal_bars=horizontal_bars,
                                            seperate_baseline_label=seperate_baseline_label,
                                            make_space_between_axis_and_bars=make_space_between_axis_and_bars, draw_norm_line=draw_norm_line,
                                            percentage=percentage,
                                            plot_average=plot_average,
                                            min_y_lim=min_y_lim,
                                            hatch_dict = hatch_dict,
                                            sub_plot_params=sub_plot_params,
                                            highlight_group = highlight_group,
                                            custom_bars_text = custom_bars_text,
                                            legend_n_col=legend_n_col,
                                            text_rotation=text_rotation,
                                            plot_legend = plot_legend,
                                            show_only_best_label = show_only_best_label)
    if sub_plot_params is None:
        print('#####', save_path + '{}.png'.format(plot_file_name))
        if horizontal_bars:
            plt.gca().invert_yaxis()
        
        plt.savefig(save_path + '{}.png'.format(plot_file_name), format='png',
                    bbox_inches='tight', dpi=dpi_val)
        plt.savefig(save_path + '{}.pdf'.format(plot_file_name), format='pdf',
                    bbox_inches='tight', dpi=dpi_val)
        json_obj = json.dumps(data_dict)
        with open(save_path + '{}.json'.format(plot_file_name.lower()), 'w') as f:
            f.write(json_obj)
        plt.clf()
        plt.close()


def bar_mapping_groups(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels, plot_name, board_name, model_name,
                       y_axis_unit, plot_mode='sep', normalized=True, normalize=False):

    save_path = os.getcwd() + '/figures/{}/'.format(board_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += metric + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += model_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    plt.figure(figsize=(4, 2))

    plot_dict = prepare_plot_dict(
        xpoints_grp, ypoints_grp, series_labels, normalized, normalize)
    if plot_mode == 'sep':
        separate_bars(xpoints_grp, plot_dict)
    elif plot_mode == 'inter':
        plot_bars_as_interleaved_bar_groups(len(xpoints_grp[0]), plot_dict)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel(axis_labels[0])
    y_axis_label_postfix = ' normalized' if normalized else '(' + \
        y_axis_unit + ')'
    plt.ylabel(axis_labels[1] + y_axis_label_postfix)
    plt.title(board_name + '_' + model_name)
    # ax.set_xticks(x + width, xpoints_grp[0])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, frameon=False, borderpad=0, columnspacing=0.4)
    plot_name_postfix = 'normalized' if normalized else y_axis_unit
    plt.savefig(save_path + '{}_{}.png'.format(plot_name,
                plot_name_postfix), bbox_inches='tight')
    plt.clf()
    plt.close()


def bar_model_groups(xpoints_grp, ypoints_grp, metric, series_labels, axis_labels, plot_name, board_name, mapping_name,
                     y_axis_unit='', plot_mode='sep', normalized=True, colors=[]):

    save_path = mapping_general_utils.prepare_save_path_from_list(
        os.getcwd() + '/figures', [board_name, metric, mapping_name])

    plt.figure(figsize=(4, 2))

    plot_dict = prepare_plot_dict(
        xpoints_grp, ypoints_grp, series_labels, normalized)
    if plot_mode == 'sep':
        separate_bars(xpoints_grp, plot_dict, colors)
    elif plot_mode == 'inter':
        plot_bars_as_interleaved_bar_groups(
            len(xpoints_grp[0]), plot_dict, colors)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel(axis_labels[0])
    y_axis_label_postfix = ''
    if normalized or y_axis_unit != '':
        y_axis_label_postfix = ' normalized' if normalized else '(' + \
            y_axis_unit + ')'
    plt.ylabel(axis_labels[1] + y_axis_label_postfix)
    plt.title(board_name + '_' + mapping_name)
    # ax.set_xticks(x + width, xpoints_grp[0])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=3, frameon=False, borderpad=0, columnspacing=0.4)
    plot_name_postfix = 'normalized' if normalized else y_axis_unit
    if plot_name_postfix != '':
        plot_name = '{}_{}.png'.format(plot_name, plot_name_postfix)
    else:
        plot_name = plot_name + '.png'
    plt.savefig(save_path + plot_name, bbox_inches='tight')
    plt.clf()
    plt.close()
