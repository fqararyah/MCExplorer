from plot import plot_results
import mapping_utils.mapping_general_utils as mapping_general_utils
import constants


hm_dict = mapping_general_utils.load_json_to_dict(
    './optimizers/timing/ref/hm_max_clusters_8.json')
ga_dict = mapping_general_utils.load_json_to_dict(
    './optimizers/timing/ref/ga_population_400_generations_50.json')
sa_dict = mapping_general_utils.load_json_to_dict(
    './optimizers/timing/ref/sa_iterations_20.json')

hm_dict = mapping_general_utils.swap_levels(hm_dict, 0, 1)
ga_dict = mapping_general_utils.swap_levels(ga_dict, 0, 1)
sa_dict = mapping_general_utils.swap_levels(sa_dict, 0, 1)

def calc_timings(in_dict):
    model_timing_dict = {}
    for model_name, boards_dict in in_dict.items():
        model_timing_dict[model_name] = 0
        num_boards = 0
        for board_name, timing_dict in boards_dict.items():
            model_timing_dict[model_name] += timing_dict['duration']
            num_boards += 1
        model_timing_dict[model_name] /= (num_boards * 60) #second to minute

    return model_timing_dict

hm_model_timing_dict = calc_timings(hm_dict)
ga_model_timing_dict = calc_timings(ga_dict)
sa_model_timing_dict = calc_timings(sa_dict)

hm_model_timing_dict_norm_to_sa = {}
hm_model_timing_dict_norm_to_ga = {}
# for key, val in ga_model_timing_dict.items():
#     hm_model_timing_dict_norm_to_sa[key] =  sa_model_timing_dict[key] / val
#     hm_model_timing_dict_norm_to_ga[key] = ga_model_timing_dict[key] / val

print(sa_model_timing_dict)
print(ga_model_timing_dict)
print(hm_model_timing_dict)