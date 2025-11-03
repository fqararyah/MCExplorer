import os
from preformance_record import Metrics
from hw_config import *

TEST_SEEDS = True
ENABLE_LOGGING = False
LOGGING_PATH = '../logs'

this_dir = os.path.dirname(__file__) 

MODEL_ARCH_DIR = this_dir + '/models_archs/'
CLEANED_RESULTS_DIR = '../cleaned_results'

PAPER_EXPERIMENTS_V = 2
FIGURES_DIR = os.path.dirname(__file__) + '/figures'
FIGURES_DATA_DIR = os.path.dirname(__file__) + '/figures_data'

FIGURES_DIR_P2 = os.path.dirname(__file__) + '/figures_p2'
FIGURES_DATA_DIR_P2 = os.path.dirname(__file__) + '/figures_data_p2'

HW_CONFIGS_FILE_v1 = os.path.dirname(__file__) + '/config_files/hw_configs.json'
HW_CONFIGS_FILE = os.path.dirname(__file__) + '/config_files/hw_configs_v2.json'
HW_CONFIGS_FILE_PEs_LARGE_MEM = os.path.dirname(__file__) + '/config_files/hw_configs_v2_pes_large_mem.json'
HW_CONFIGS_FILE_PEs_LIMITED_MEM = os.path.dirname(__file__) + '/config_files/hw_configs_v2_pes_limited_mem.json'
HW_CONFIGS_FILES_v2 = [HW_CONFIGS_FILE, HW_CONFIGS_FILE_PEs_LARGE_MEM, HW_CONFIGS_FILE_PEs_LIMITED_MEM, HW_CONFIGS_FILE_v1]
CUSTOM_MAPPINGS_JSON_DIR = os.path.dirname(__file__) + '/custom_mappings_json'
CUSTOM_MAPPING_FILE = 'cutom_mappings.json'
POWER_FILES_PATH = os.path.dirname(__file__) + '/power_energy'

BIT_WIDTH = 8
GiB = 2 ** 30
GB = 10 ** 9
MiB = 2 ** 20
KiB = 1024

BRAM_BLOCK_KBITS = 18
BRAM_BLOCK_BYTES = 18 * KiB / 8

MIN_ENGINES = 2
MAX_ENGINES = 11

MIN_ENGINES_V2 = 2
MIN_SEGMENTS = 1
MAX_ENGINES_V2 = 11
MAX_CLUSTERS = 8
MIN_CLUSTERS = 2

model_names = ['unet','resnet152', 'resnet50', 'xce_r', 'dense121', 'mob_v2', 'cmt']
mappings_ordered = ['Hybrid', 'Segmented', 'SegmentedRR']
model_display_names = {'resnet50': 'Res50', 'resnet152': 'Res152',
                       'mob_v2': 'MobV2', 'dense121': 'Dns121', 'xce_r': 'XCp', 'unet': 'SmAt-U', 'cmt': 'CMT'}

metric_list = [Metrics.LATENCY, Metrics.THROUGHPUT,
                   Metrics.ACCESS, Metrics.BUFFER]

metric_display_names = {Metrics.ACCESS: 'Access', Metrics.BUFFER: 'Buffers',
                        Metrics.LATENCY: 'Latency', Metrics.THROUGHPUT: 'Throughput',
                        Metrics.ENERGY: 'Energy'}

metric_file_names = {Metrics.ACCESS: 'acc', Metrics.BUFFER: 'buff',
                        Metrics.LATENCY: 'lat', Metrics.THROUGHPUT: 'thr',
                        Metrics.ENERGY: 'energy'}

display_names_metrics_dict = {'Access': Metrics.ACCESS, 'Buffers': Metrics.BUFFER,
                        'Latency': Metrics.LATENCY, 'Throughput': Metrics.THROUGHPUT,
                        'Energy': Metrics.ENERGY}