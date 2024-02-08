import numpy as np
import pandas as pd 
import os
from fnmatch import fnmatch
from pathlib import Path

import igor2 as igor
from igor.packed import load as loadpxp
from igor.record.wave import WaveRecord

from tkinter import simpledialog

from Intracellular_analysis_functions import load_NM_data_create_sweeps, create_traces_and_exclude 

google_drive_path = simpledialog.askstring("Google Drive Path", "Enter a directory Path:")
google_drive_path = Path(google_drive_path)

dates = simpledialog.askstring("Experiment Date", "Enter a date (mmddyyyy):")
dates = [dates]

experiment_subfolder = simpledialog.askstring("Experiment Subfolder", "Enter the name of the experiment subfolder:")
cell = simpledialog.askstring("Cell Number", "Enter the cell number:")

stimulation_protocols = [b'IV_stim', b'EPSP_stim'] 
path = google_drive_path
data_df = pd.DataFrame(columns=['experiment_index', 'date', 'dir_name', 'file_name', 'cell_number', 'sweep_numbers', 'traces','stim_traces'])
metadata_df = pd.DataFrame(columns=['experiment_index', 'file_name' ,'cell_number',	'stim_type','date',	'inter_trial_interval',	'samples_per_wave',	'experiment_start_time','experiment_end_time',
                                    'total_experiment_time (s)',	'total_experiment_window (ms)',	'dt',	'acquisition_frequency'])


dict_list_data = []
dict_list_metadata = []
exclude_trace_list =  []

for date in dates:
    myDir = Path(f'{google_drive_path}/{experiment_subfolder}/CA1_Patch_{date}') 

    fileNames = [file.name for file in myDir.iterdir() if file.name.startswith('Record_CA1') and file.name.endswith(f'{cell}')]    
    filePaths = [file for file in myDir.iterdir() if file.name.startswith('Record_CA1') and file.name.endswith(f'{cell}')]

    for path in filePaths:
        root = path

        print(f'Loading data from {root}')
        file_pattern = "*.pxp"

        for path, subdirs, files in os.walk(root):
            for i, name in enumerate(files):
                if fnmatch(name, file_pattern):
                    cell_file_path = os.path.join(path, name)


                    row_dict_data = {'experiment_index': name[-7:-4], 'date': date, 'dir_name': cell_file_path, 'file_name': name}
                    row_dict_metadata = {'experiment_index': name[-7:-4], 'file_name': name} 



                    row_dict_data['cell_number'] = cell 
                    row_dict_metadata['cell_number'] = cell

                dict_list_data.append(row_dict_data)
                dict_list_metadata.append(row_dict_metadata)

                for stimulation_set in stimulation_protocols:
                    if stimulation_set == b'IV_stim':
                        try:
                            records_dict, waves_dict, root_waves_dict, experiment_metadata, dt, acquisition_frequency, sweeps, stims =  load_NM_data_create_sweeps(cell_file_path, stimulation_set) 

                            raw_traces, sweep_numbers, excluded_sweep_numbers, excluded_traces = create_traces_and_exclude(sweeps, exclude_trace_list)

                            row_dict_data['sweep_numbers'] = sweep_numbers
                            row_dict_data['traces'] = raw_traces
                            row_dict_data['stim_traces'] = stims

                           #Fill in metadata
                            row_dict_metadata['stim_type'] = stimulation_set
                            row_dict_metadata['date'] = experiment_metadata['date']
                            row_dict_metadata['inter_trial_interval'] = experiment_metadata['inter trial interval (ms)']
                            row_dict_metadata['samples_per_wave'] = experiment_metadata['samples per wave']
                            row_dict_metadata['experiment_start_time'] = experiment_metadata['experiment start time']
                            row_dict_metadata['experiment_end_time'] = experiment_metadata['experiment end time']
                            row_dict_metadata['total_experiment_time (s)'] = experiment_metadata['total experiment time (s)']
                            row_dict_metadata['total_experiment_window (ms)'] = experiment_metadata['total recording window (ms)'] 
                            
                            row_dict_metadata['dt'] = dt
                            row_dict_metadata['acquisition_frequency'] = acquisition_frequency


                        except:
                            pass
                            # print(f'Error loading {cell_file_path}')

                    elif stimulation_set == b'EPSP_stim':
                        try:
                            records_dict, waves_dict, root_waves_dict, experiment_metadata, dt, acquisition_frequency, sweeps, stims =  load_NM_data_create_sweeps(cell_file_path, stimulation_set)

                            raw_traces, sweep_numbers, excluded_sweep_numbers, excluded_traces = create_traces_and_exclude(sweeps, exclude_trace_list)

                            row_dict_data['sweep_numbers'] = sweep_numbers
                            row_dict_data['traces'] = raw_traces
                            row_dict_data['stim_traces'] = stims

                            #Fill in metadata
                            row_dict_metadata['stim_type'] = stimulation_set
                            row_dict_metadata['date'] = experiment_metadata['date']
                            row_dict_metadata['inter_trial_interval'] = experiment_metadata['inter trial interval (ms)']
                            row_dict_metadata['samples_per_wave'] = experiment_metadata['samples per wave']
                            row_dict_metadata['experiment_start_time'] = experiment_metadata['experiment start time']
                            row_dict_metadata['experiment_end_time'] = experiment_metadata['experiment end time']
                            row_dict_metadata['total_experiment_time (s)'] = experiment_metadata['total experiment time (s)']
                            row_dict_metadata['total_experiment_window (ms)'] = experiment_metadata['total recording window (ms)'] 
                            
                            row_dict_metadata['dt'] = dt
                            row_dict_metadata['acquisition_frequency'] = acquisition_frequency
                        except:
                            pass
                            # print(f'Error loading {cell_file_path}')


    
data_df = pd.DataFrame.from_dict(dict_list_data) 
metadata_df = pd.DataFrame.from_dict(dict_list_metadata) 

data_pickle_name = simpledialog.askstring("Data Pickle File", "Enter a name (date_cell):")
metadata_pickle_name = simpledialog.askstring("Metadata Pickle File", "Enter a name (date_cell):")

data_df.to_pickle(f'{google_drive_path}/{data_pickle_name}.pkl')
metadata_df.to_pickle(f'{google_drive_path}/{metadata_pickle_name}.pkl')