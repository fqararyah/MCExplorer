
from plot import plot_results
import mapping_utils.mapping_general_utils as mapping_general_utils
import constants

json_file = constants.FIGURES_DATA_DIR_P2 + '/validation_thr_lat_enr.json'

estimated_actual_dict = mapping_general_utils.load_json_to_dict(json_file)
accuracies = []

#for key, val in estimated_actual_dict.items():
estimated_vals_list = estimated_actual_dict['estimated']
actual_vals_list = estimated_actual_dict['actual']
for i, estimated_val in enumerate(estimated_vals_list):
    accuracies.append( 1 - abs(estimated_val - actual_vals_list[i]) / actual_vals_list[i] )

print(accuracies)
accuraices_dict = {'Accuracy': accuracies}

plot_results.plot_bar_groups(accuraices_dict, 'mcexplorer_validation',
                                 x_title=None,
                                 y_title='accuracy',
                                 x_ticks_dict={1:['Throughput', 'Latency', 'Energy']}, relative_save_path='validation',
                                 abs_save_path=constants.FIGURES_DIR_P2)