import __init__
import mapping_utils.mapping_general_utils as mapping_general_utils
from hw_config import *
import constants as consts
import os


for board_name in os.listdir(consts.POWER_FILES_PATH):
    file_path = consts.POWER_FILES_PATH + '/' + board_name + '/'
    if not os.path.isdir(file_path):
        continue
    for pow_file in os.listdir(file_path):
        if 'csv' in pow_file:
            ret_dict = mapping_general_utils.csv_to_dict(file_path + pow_file)
            x = ret_dict['units']
            y = list(ret_dict['power'])
            resource = None
            if 'dsp' in pow_file:
                resource = Resource.DSP
            elif 'bram' in pow_file:
                resource = Resource.BRAM
                if 'write' in pow_file:
                    resource = Resource.BRAM_W
                elif 'read' in pow_file:
                    resource = Resource.BRAM_R
                else:
                    resource = Resource.BRAM
            if pow_file.split('.')[0].endswith('_d'):
                idle = False
            else:
                idle = True
            mapping_general_utils.approximate_power_v2(1, board_name, resource , idle, ret_dict, calculate_anyway = True)
