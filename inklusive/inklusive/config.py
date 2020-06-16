# config.py
# Special paths and variables used across the project
import os

path_project = '/Users/brynronalds/Insight/proj_dir/'
path_models = os.path.join(path_project,'models')
path_data = os.path.join(path_project,'data')
path_data_raw = os.path.join(path_data,'raw')
path_data_proc = os.path.join(path_data,'processed')
path_data_cln = os.path.join(path_data,'cleaned')
path_data_upld = os.path.join(path_data, 'upload')

path_tattoodf = os.path.join(path_data_proc,'Inklusive_database','train')
tattoodf_name = os.path.join(path_tattoodf,'tattoo_info.csv')
