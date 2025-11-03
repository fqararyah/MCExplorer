from mapping_types.hybrid_mapping import *
from mapping_types.segment_grained_mapping_rr import *
from mapping_types.segment_grained_mapping import *
from preformance_record import *
import experiments_p2.experiments_utils as experiment_utils
from mapping_strategies.mapping_types.mapping_description import *
import mapping_utils.custom_mapping_utils as custom_mapping_utils
import ast

def heterogeneity_counts(in_dict, metric):
    heterogeneity_counts_list = [0] * 4
    for metric_label, metric_dict in in_dict.items():
        for board_name, board_name_dict in metric_dict.items():
            for model_name, model_name_dict in board_name_dict.items():
                model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                mapping_desc_dict = model_name_dict[1]
                mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
                    board_name, model_dag, mapping_desc_dict,
                    timing_metric=Metrics.LATENCY if metric == Metrics.LATENCY else Metrics.THROUGHPUT,
                    adjust_pes=True, enable_multi_ce=True)

                has_heterogeneous_blocks = mapping.has_heterogeneous_blocks()
                has_heterogeneous_parallelism_strategies = mapping.has_heterogeneous_parallelism_strategies()
                if has_heterogeneous_blocks and has_heterogeneous_parallelism_strategies:
                    heterogeneity_counts_list[3] += 1
                elif has_heterogeneous_blocks:
                    heterogeneity_counts_list[2] += 1
                elif has_heterogeneous_parallelism_strategies:
                    heterogeneity_counts_list[1] += 1
                else:
                    heterogeneity_counts_list[0] += 1

    return heterogeneity_counts_list

def intra_ce_heterogeneity_counts(population = 400,
                    generations = 50,
                    sa_tenthousands = 20000,
                    hm_max_clusters = constants.MAX_CLUSTERS):
    heterogeneity_counts_dict = {}

    metrics = [Metrics.THROUGHPUT, Metrics.LATENCY, Metrics.ENERGY]
    for metric in metrics:
        black_box_dic = experiment_utils.get_best_mappings_black_box(metric,
                                                             population, generations, sa_tenthousands, hm_max_clusters)
        tmp_blocks_labels = black_box_dic['MCExplorer'][constants.metric_display_names[metric]]
        for board_name, board_name_dict in tmp_blocks_labels.items():
            for model_name, model_name_dict in board_name_dict.items():
                model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                mapping_desc_dict = model_name_dict[1]
                mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
                    board_name, model_dag, mapping_desc_dict,
                    timing_metric=Metrics.LATENCY if metric == Metrics.LATENCY else Metrics.THROUGHPUT,
                    adjust_pes=True, enable_multi_ce=True)
                
                par_strategies = str(mapping.get_parallelism_strategies())
                min_dfs = str(mapping.get_minmum_df_possibilities())
                if par_strategies not in heterogeneity_counts_dict:
                    heterogeneity_counts_dict[par_strategies] = {}
                if min_dfs not in heterogeneity_counts_dict[par_strategies]:
                    heterogeneity_counts_dict[par_strategies][min_dfs] = 0
                
                heterogeneity_counts_dict[par_strategies][min_dfs] += 1


    a_table = []
    unique_dfs_dict = {}
    unique_strategies_dict = {}
    df_index = 0
    sum_all = 0
    for strategy, dfs_dict in heterogeneity_counts_dict.items():
        a_table.append([0] * 5)
        print(strategy)
        for df, cont in dfs_dict.items():
            lst = ast.literal_eval(df)
            for sub_df in lst:
                if sub_df not in unique_dfs_dict:
                    unique_dfs_dict[sub_df] = df_index
                    df_index += 1
                a_table[-1][unique_dfs_dict[sub_df]] += cont
                sum_all += cont

    print(unique_strategies_dict)
    print(unique_dfs_dict)
    row_num = 0
    unused = 0
    for row in a_table:
        row_str = 'parallelsim strategy {}: '.format(row_num)
        header_str = ' ' * len(row_str)
        for i in range(5):
            header_str += 'DF_{}&   '.format(i)
        if row_num == 0:
            print(header_str)
        row_num += 1
        for col in row:
            row_str += str(round(col / sum_all, 2)) + '&    '
            if col == 0:
                unused += 1
        print(row_str)
        print('UNUSED: ', unused)
    return heterogeneity_counts_dict
                    

def block_types(in_dict, metric):
    block_types_dict = {}
    for metric_label, metric_dict in in_dict.items():
        for board_name, board_name_dict in metric_dict.items():
            for model_name, model_name_dict in board_name_dict.items():
                if model_name not in block_types_dict:
                    block_types_dict[model_name] = {}
                model_dag = utils.read_model_dag_v2(
                    constants.MODEL_ARCH_DIR + model_name + '/model_dag.json')
                mapping_desc_dict = model_name_dict[1]
                mapping = custom_mapping_utils.custom_mapping_from_desc_dict(
                    board_name, model_dag, mapping_desc_dict,
                    timing_metric=Metrics.LATENCY if metric == Metrics.LATENCY else Metrics.THROUGHPUT,
                    adjust_pes=True, enable_multi_ce=True)
                block_repr = mapping.block_labels_repr()
                if block_repr not in block_types_dict[model_name]:
                    block_types_dict[model_name][block_repr] = 0
                block_types_dict[model_name][block_repr] += 1
                
    return block_types_dict

def hereto_blocks_table(population = 400,
                    generations = 50,
                    sa_tenthousands = 20000,
                    hm_max_clusters = constants.MAX_CLUSTERS):
    metrics = [Metrics.THROUGHPUT, Metrics.LATENCY, Metrics.ENERGY]
    blocks_labels = {}
    num_exps = 0
    for metric in metrics:
        black_box_dic = experiment_utils.get_best_mappings_black_box(metric,
                                                             population, generations, sa_tenthousands, hm_max_clusters)
        tmp_blocks_labels = block_types(black_box_dic['MCExplorer'], metric)
        for model_name in list(tmp_blocks_labels.keys()):
            if model_name not in blocks_labels:
                blocks_labels[model_name] = {}
            for key, val in tmp_blocks_labels[model_name].items():
                if key not in blocks_labels[model_name]:
                    blocks_labels[model_name][key] = 0
                blocks_labels[model_name][key] += val
                num_exps += val
        
    table_lines = ['\\rowcolor{gray!10} Model & ', '\cellcolor{gray!10} CE-blocks & ', '\cellcolor{gray!10} Experiments & ' ]
    all_blocks = 0
    for model_name, block_dict in list(blocks_labels.items()):
        table_lines[0] += '\multicolumn{' + str(len(block_dict)) + '}{c|}{' + constants.model_display_names[model_name] +  '}& '
        for key, val in block_dict.items():
            table_lines[1] += key + '& '
            table_lines[2] += str(val) + '& '
            all_blocks += 1
    
    for i in range(len(table_lines)):
        table_lines[i] = table_lines[i].strip('& ') + '\\\\'

    table_lines.insert(0, '\\begin{tabular}{|l|' + 'c|' * (all_blocks) + '}')
    table_lines.insert(1, '\hline')
    table_lines.insert(3, '\hline')
    table_lines.append('\hline')
    for line in table_lines:
        print(line)
        
population = 400
generations = 50
sa_tenthousands = 20000
hm_max_clusters = constants.MAX_CLUSTERS

intra_ce_dict = intra_ce_heterogeneity_counts()
print(intra_ce_dict)

metric = Metrics.THROUGHPUT
black_box_dic = experiment_utils.get_best_mappings_black_box(metric,
                                                             population, generations, sa_tenthousands, hm_max_clusters)
heterogeneity_counts_list_th = heterogeneity_counts(
    black_box_dic['MCExplorer'], metric)
print(heterogeneity_counts_list_th)

metric = Metrics.LATENCY
black_box_dic = experiment_utils.get_best_mappings_black_box(metric,
                                                             population, generations, sa_tenthousands, hm_max_clusters)
heterogeneity_counts_list_lat = heterogeneity_counts(
    black_box_dic['MCExplorer'], metric)
print(black_box_dic)
print(heterogeneity_counts_list_lat)

metric = Metrics.ENERGY
black_box_dic = experiment_utils.get_best_mappings_black_box(metric,
                                                             population, generations, sa_tenthousands, hm_max_clusters)
heterogeneity_counts_list_en = heterogeneity_counts(
    black_box_dic['MCExplorer'], metric)
    
print(heterogeneity_counts_list_en)

heterogeneity_counts_list = []
for i in range(len(heterogeneity_counts_list_th)):
    heterogeneity_counts_list.append(
        heterogeneity_counts_list_th[i] + heterogeneity_counts_list_lat[i] + heterogeneity_counts_list_en[i])

sum_heterogeneity_counts_list = sum(heterogeneity_counts_list)
heterogeneity_counts_list = [heterogeneity_counts_list[i] / sum_heterogeneity_counts_list for i in range(len(heterogeneity_counts_list))] 
print(heterogeneity_counts_list)

hereto_blocks_table()

