import numpy as np
import pandas as pd 
import os
import re
from fnmatch import fnmatch
from pathlib import Path
import matplotlib.pyplot as plt

import igor2 as igor
from igor.packed import load as loadpxp
from igor.record.wave import WaveRecord

from tkinter import simpledialog

from Intracellular_analysis_functions_05032024 import load_NM_data_create_sweeps 
''' Make sure original data files do not have dates on the end of them and are named in the following format:
'Record......c1_001 for example no date at the end   This is important for the code to work properly, the folder should also not 
contain any backup data or log0 data files''' 

#conda activate /opt/anaconda3/envs/Ephys_Analysis 

#Tell the user to enter the directory path for an individual cell with a given date that will contain multiple experiments
myDir = simpledialog.askstring("File Path", "Enter a directory Path (Please make sure no log0.pxp and backup.pxp files exist and that files end in index not date):")
config_file_path = '/Users/samgritz/Desktop/Rutgers/Milstein_Lab/Code/Rutgers-Neuroscience-PhD/Ephys_Analysis/Updated_Code/Intracellular_Analysis/dataframe_defaults.yaml'
#Now get the .pxp files from this directory and load those files
#Generalized code to get out files
pattern = "*.pxp"

pxp_files = []
for path, subdirs, files in os.walk(myDir):
    # row_dict_data = {'experiment_index':[], 'dir_name':[], 'file_name':[], 'cell_number':[]}
    for name in files:
        # row_dict_data['experiment_index'].append(name[-6:-4])
        if fnmatch(name, pattern):
            current_file = os.path.join(path, name)
            pxp_files.append(current_file)

index_list = []
for file in pxp_files:
    index = str(file[-7:-4])
    index_list.append(index)
#Sort Index List
sorted_index_list = sorted(index_list, key=int, reverse=False)

#Now append the name of the file to the sorted_files list
sorted_files = []
for index in sorted_index_list:
    # print(f'{file[1:-7]}{index}.pxp') #Prints the file name without the .pxp
    #recreate the file name to match pxp file
    current_sorted_file = f'/{file[1:-7]}{index}.pxp'
    sorted_files.append(current_sorted_file)

master_df = pd.DataFrame()
for file in sorted_files:
    sweeps, stim_records, stim_commands, acquisition_metadata_dict, stimulus_metadata_dict, analysis_dict =  load_NM_data_create_sweeps(file, yaml_file_path=config_file_path)
    print(acquisition_metadata_dict)
    for index, sweep in enumerate(sweeps):
        if index == 0:
            initial_global_wall_clock = acquisition_metadata_dict['global wall clock']
        sweep_duration = acquisition_metadata_dict['samples per wave'] * acquisition_metadata_dict['dt'] / 1000  # seconds
        inter_trial_delay = acquisition_metadata_dict['inter trial interval (ms)'] / 1000  # seconds
        current_global_wall_clock = initial_global_wall_clock + index * (sweep_duration + inter_trial_delay)
        dict_metadata = {}

        try:
            dict_metadata['experiment_index'] = int(file[-7:-4])
            dict_metadata['sweep_index'] = index
            dict_metadata['cell_number'] = file[-10:-8]
            dict_metadata['file_path']= file
            dict_metadata['stim_type'] = acquisition_metadata_dict['stim type']
            dict_metadata['experiment_description'] = acquisition_metadata_dict['experiment description']
            dict_metadata['date'] = acquisition_metadata_dict['date']
            dict_metadata['dt'] = acquisition_metadata_dict['dt']
            dict_metadata['acquisition_frequency'] = acquisition_metadata_dict['acquisition_frequency']
            dict_metadata['sweep_duration'] = acquisition_metadata_dict['total recording window (ms)']
            dict_metadata['total_experiment_time'] = acquisition_metadata_dict['total experiment time (s)']
            dict_metadata['inter_trial_interval'] = acquisition_metadata_dict['inter trial interval (ms)']
            dict_metadata['samples_per_wave'] = acquisition_metadata_dict['samples per wave']
            dict_metadata['global_wall_clock'] = current_global_wall_clock
            dict_metadata['experiment_start_time'] = acquisition_metadata_dict['experiment start time']
            dict_metadata['experiment_end_time'] = acquisition_metadata_dict['experiment end time']
            dict_metadata['sweep'] = sweep
            if acquisition_metadata_dict['stim type'] != 'EPSP_stim':
                dict_metadata['stim_record'] = stim_records[index]
                dict_metadata['stim_command'] = stim_commands[index]
            else:
                dict_metadata['stim_record'] = stim_records
                dict_metadata['stim_command'] = stim_commands
            dict_metadata['analysis_dict'] = analysis_dict
            dict_metadata['stimulus_metadata_dict'] = stimulus_metadata_dict
        except:
            print(f'Error with {file} sweep {index}')
            continue
   
    
      
        
        this_df_row = pd.DataFrame([dict_metadata])
        master_df = pd.concat([master_df, this_df_row], ignore_index=True)

data_pickle_name = simpledialog.askstring("Data Pickle File", "Enter a name (date_cell):") 
save_path = simpledialog.askstring("Save Path", "Enter a save path (e.g. /content/drive/My Drive/):")
master_df.to_pickle(f'{save_path}/{data_pickle_name}.pkl')



