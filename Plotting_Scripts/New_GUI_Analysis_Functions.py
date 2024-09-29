import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.figure as figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import scipy
from pkg_resources import non_empty_lines
# import scipy.stats as stats
from scipy.signal import find_peaks

# import statsmodels
# from statsmodels.stats.multitest import multipletests
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import AnovaRM

# Set matplotlib configurations
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 12.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False

'''Functions for plotting AP properties, intrinsic properties, and firing rates'''

def convert_to_csv(data_dict, save_path, save_name):
    fine_FI_stim_properties_df = pd.DataFrame(data_dict)
    fine_FI_stim_properties_df.to_csv(save_path + '/' + save_name)

def get_intrinsic_properties_cells(data_dir, cell_properties_to_plot):
    data_files = os.listdir(data_dir)
    data_files = [file for file in data_files if file.endswith('.pkl')]

    intrinsic_cell_properties = {}
    for data_file in data_files:
        data_file_date = data_file.split('.')[0]
        data_file_cell = data_file_date.split('_')[1]
        data_file_name = data_file_cell + '_' + data_file_date
        current_data_df = pd.read_pickle(os.path.join(data_dir, data_file))
            
        intrinsic_cell_properties[data_file_name] = {}

        for i in range(len(current_data_df)):
            if 'Intrinsic_cell' in current_data_df['analysis_dict'][i]:
                if current_data_df['analysis_dict'][i]['Intrinsic_cell'] is not None:
                    current_intrinsics = current_data_df['analysis_dict'][i]['Intrinsic_cell']
                    # print(current_intrinsics, data_file_name)
                        
                    for key in cell_properties_to_plot:
                        if key not in intrinsic_cell_properties[data_file_name]:
                            intrinsic_cell_properties[data_file_name][key] = []

                        # Append the value only if it's not None
                        if key in current_intrinsics and current_intrinsics[key] is not None:
                            intrinsic_cell_properties[data_file_name][key].append(current_intrinsics[key])
                        else:
                            # Append NaN if the key is missing or the value is None
                            intrinsic_cell_properties[data_file_name][key].append(np.nan)

    try:
        # Average across the properties for each cell
        average_intrinsic_cell_properties = {}

        for cell in intrinsic_cell_properties:
            average_intrinsic_cell_properties[cell] = {}
            for key in intrinsic_cell_properties[cell]:
                # Calculate the mean ignoring NaN values
                average_intrinsic_cell_properties[cell][key] = np.nanmean(intrinsic_cell_properties[cell][key])

        return average_intrinsic_cell_properties
    
    except Exception as e:
        print('Error in extracting intrinsic properties:', str(e))
        return None

#create a function that takes in a list of sweep numbers and dates and a folder of files and it will go through each file that has that name and populate a dictionary of AP properties
#get AP properties based on a list of sweep numbers and cell names

def get_AP_properties_data(data_dir, sweep_list, AP_properties_to_plot):

    sweep_list = sweep_list

    data_files = os.listdir(data_dir)
    data_files = [file for file in data_files if file.endswith('.pkl')]
    AP_properties_to_plot = AP_properties_to_plot

    fine_FI_stim_properties = {}
    sorted_data_files = []
    for data_file in data_files:
        # Read in the data file
        data_df = pd.read_pickle(data_dir + '/' + data_file)
        # Get the name of the cell - everything in the file name except '_processed_new.pkl'
        cell_date = data_file.split('_')[0]
        cell_name = data_file.split('_')[1]
        data_name = cell_name + '_' + cell_date
        # Find the date pattern of the data_name and sort the data by date
        date = data_name.split('_')[1]
        sorted_data_files.append((date, data_name, data_df))
        
    sorted_data_files.sort()

    try:
        for idx, sweep in enumerate(sweep_list):
            current_analysis_data = sorted_data_files[idx][2]['analysis_dict'][sweep]
            data_name = sorted_data_files[idx][1]
            # print(data_name)
            current_analysis_data_AP = sorted_data_files[idx][2]['analysis_dict'][sweep]['AP']

            if data_name not in fine_FI_stim_properties:
                fine_FI_stim_properties[data_name] = {}
                
            for key in AP_properties_to_plot:
                if key not in fine_FI_stim_properties[data_name]:
                    fine_FI_stim_properties[data_name][key] = []
                
                # Append the first value if the length of the data is greater than 1
                if len(current_analysis_data_AP[key]) > 1:
                    fine_FI_stim_properties[data_name][key].append(current_analysis_data_AP[key][0])
                else:
                    fine_FI_stim_properties[data_name][key].append(current_analysis_data_AP[key])
            
            try: 
                # Initialize 'Rheobase_Current' if 'Fine_FI', 'IV_stim', or 'Coarse_FI' is in current_analysis_data
                if 'Fine_FI' in current_analysis_data: #or 'IV_stim' in current_analysis_data or 'Coarse_FI' in current_analysis_data:
                    if 'Rheobase_Current' not in fine_FI_stim_properties[data_name]:
                        fine_FI_stim_properties[data_name]['Rheobase_Current'] = []
                    
                    if 'Fine_FI' in current_analysis_data and 'current_amplitudes' in current_analysis_data['Fine_FI']:
                        fine_FI_stim_properties[data_name]['Rheobase_Current'].append(current_analysis_data['Fine_FI']['current_amplitudes'])

                    # elif 'IV_stim' in current_analysis_data and 'current_amplitudes' in current_analysis_data['IV_stim']:
                    #     fine_FI_stim_properties[data_name]['Rheobase_Current'].append(current_analysis_data['IV_stim']['current_amplitudes'])
                    # elif 'Coarse_FI' in current_analysis_data and 'current_amplitudes' in current_analysis_data['Coarse_FI']:
                    #     fine_FI_stim_properties[data_name]['Rheobase_Current'].append(current_analysis_data['Coarse_FI']['current_amplitudes'])
            except Exception as e:
                print(f'Error in extracting rheobase current for cell {data_name}:', str(e))
                pass
        
        return fine_FI_stim_properties


    except Exception as e:
        print(f'Error in extracting AP properties for cell {data_name}:', str(e))
        return None

def get_fine_AP_properties(data_dict, AP_properties_to_plot):
    AP_properties_dict = {}
    for cell in data_dict:
        for key in AP_properties_to_plot:
            if key not in AP_properties_dict:
                AP_properties_dict[key] = []
            AP_properties_dict[key].extend(data_dict[cell][key])
    return AP_properties_dict

# Function to remove NaN values and flatten nested lists
def clean_data(values):
    flat_list = []
    for value in values:
        if isinstance(value, list):
            flat_list.extend(value)
        else:
            flat_list.append(value)
    return [value for value in flat_list if not (isinstance(value, float) and np.isnan(value))]

def get_mean_std_error(data_dict, AP_properties_to_plot, bar_plot_keys):
    mean_dict = {}
    std_error_dict = {}

    values_list = {}
    for cell in data_dict:
        for idx, key in enumerate(AP_properties_to_plot):
            for values in data_dict[cell][key]:
                current_values = []
                #if the value is not nan and greater than zero then append
                if not np.isnan(values).all():
                    if key in bar_plot_keys:
                        if isinstance(values, list):
                            if values[0] > 0:
                                current_values.extend(values)
                        else:
                            if values > 0:
                                current_values.append(values)
                    else:
                        current_values.append(values)

                if key not in values_list:
                    values_list[key] = []

                values_list[key].extend(current_values)
        
    #now get the mean and std error for the values
    mean_dict = {}
    std_error_dict = {}

    for key in AP_properties_to_plot:
        data = clean_data(values_list[key])
        mean_dict[key] = np.mean(data)
        std_error_dict[key] = np.std(data)/np.sqrt(len(data))

    return mean_dict, std_error_dict


def create_dict_of_lists(data_dict):
    data_dict_list = {}
    for key in data_dict:
        for value in data_dict[key]:
            if key not in data_dict_list:
                data_dict_list[key] = []
            
            if isinstance(value, list):
                data_dict_list[key].extend(value)
            else:
                data_dict_list[key].append(value)
    return data_dict_list

'''Extract FI data from the data_df dataframe and create a table of F_I data per cell'''

def get_holding_potentials(dir_path):
    holding_potentials = {}
    pkl_files = []

    # Collect all pickle files from the directory
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.pkl'):
                pkl_files.append(os.path.join(path, name))

    # Process each pickle file
    for file in pkl_files:
        data_file = pd.read_pickle(file)
        analysis_dicts = data_file['analysis_dict']
        sweep_data = data_file['sweep']

        # Extract cell name and number
        cell_name, cell_number = os.path.basename(file).split('_')[:2]
        full_cell_name = f"{cell_name}_{cell_number}"

        if full_cell_name not in holding_potentials:
            holding_potentials[full_cell_name] = []

        holding_potential_values = []
        # Collect Holding potentials averaged across Coarse Traces
        for i, analysis_data in enumerate(analysis_dicts):
            if 'Coarse_FI' in analysis_data:
                trace = sweep_data[i]
                baseline_voltage = np.mean(trace[:int(10 * 20000 / 1000)])
                holding_potential_values.append(baseline_voltage)
        
        holding_potential_average = np.mean(holding_potential_values) 
        holding_potentials[full_cell_name].append(holding_potential_average) 

    
    return holding_potentials 

def get_F_I_data(dir_path):
    FI_data = {}
    pkl_files = []
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.pkl'):
                pkl_files.append(os.path.join(path, name))
                cell_name = name.split('_')[0]
                cell_number = name.split('_')[1]
                full_cell_name = cell_name + '_' + cell_number
                FI_data[full_cell_name] = {}

                #open all of the pickle files and create a dataframe of their analysis_dicts
                for file in pkl_files:
                    data_file = pd.read_pickle(file)
                    analysis_data = data_file['analysis_dict']


                    coarse_f_I_stim = {}
                    for idx, analysis in enumerate(analysis_data):
                        if 'Coarse_FI' in analysis.keys():
                            #iterate through the keys in the analysis dictionary
                            #get the values per index and plot them 
                            coarse_f_I_stim[idx] = analysis['Coarse_FI']
                        elif 'IV_stim' in analysis.keys():
                            coarse_f_I_stim[idx] = analysis['IV_stim']

                    #make a tuple that is current amplitude and firing rate
                    FI_plot_data = []
                    for key, value in coarse_f_I_stim.items():
                        #get the unique values of the current amplitudes
                        current_amplitudes = np.unique(value['current_amplitudes'])
                        #round up the current amplitudes to 2 decimal places
                        current_amplitudes = np.round(current_amplitudes, 1)
                        firing_rates = value['firing_rates']
                        #get the unique values of the firing rates
                        unique_firing_rates = np.unique(firing_rates)
                        #append the current amplitudes and firing rates to the FI_plot_data list
                        FI_plot_data.append((current_amplitudes, unique_firing_rates))

                    #iterate through the FI_plot_data list and if the current amplitude is found twice average the firing rates
                    #this is to make sure that the current amplitudes are unique
                    FI_plot_data_unique = []
                    for i in range(len(FI_plot_data)):
                        current_amplitudes = FI_plot_data[i][0]
                        firing_rates = FI_plot_data[i][1]
                        for j in range(len(FI_plot_data)):
                            if i != j:
                                if np.array_equal(current_amplitudes, FI_plot_data[j][0]):
                                    avg_firing_rates = (firing_rates + FI_plot_data[j][1])/2
                                    FI_plot_data_unique.append((current_amplitudes, avg_firing_rates))
                                    break
                        else:
                            FI_plot_data_unique.append((current_amplitudes, firing_rates))

                    #sort the FI_plot_data_unique list by the current amplitudes
                    FI_plot_data_unique.sort(key=lambda x: x[0])

                    #create a dictionary of the current amplitudes and firing rates
                    FI_data[full_cell_name] = {}

                    #go through the FI_plot_data_unique list and create a dictionary of the current amplitudes and firing rates
                    #find the unique values from all of the cells and create a dictionary of the current amplitudes and firing rates
                    for i in range(len(FI_plot_data_unique)):
                        current_amplitudes = FI_plot_data_unique[i][0]
                        firing_rates = FI_plot_data_unique[i][1]
                        for j in range(len(current_amplitudes)):
                            FI_data[full_cell_name][current_amplitudes[j]] = firing_rates[j]
        
        return FI_data
                    
def calculate_input_resistance(voltage_trace, current_pulse_amp, acquisition_frequency, start_time, end_time):
        #iterate through the traces
        try:
            #convert the start, end time and acquisition frequency to integers
            start_time = int(start_time)
            end_time = int(end_time)
            acquisition_frequency = int(acquisition_frequency)
            current_pulse_amp = int(current_pulse_amp)
            # calculate the input resistance
            #convert start and end times to indices
            start_time_index = int(start_time * acquisition_frequency/1000)
            end_time_index = int(end_time * acquisition_frequency/1000)
            # #calculate the baseline voltage
            baseline_start = start_time_index - int(10 * acquisition_frequency/1000)
            vm_baseline = np.mean(voltage_trace[baseline_start:start_time_index])
            # #calculate the end voltage
            end_vm = np.mean(voltage_trace[end_time_index - baseline_start:end_time_index])
            delta_voltage = (end_vm - vm_baseline)/1000 #convert to volts
            delta_current = current_pulse_amp * 10**-12 #convert to amps
            input_resistance = abs(delta_voltage / -delta_current) * 10**-6 #convert to megaohm
            return input_resistance
        except Exception as e:
            print("Error calculating input resistance:", e)
            pass
        
def plot_F_I_data_average_multiple_cells(FI_data, plot_title, saved_path):
    #get the average firing rates of keys that are the same
    average_firing_rates = {}
    std_error_firing_rates = {}
    for cell in FI_data.keys():
        for current_amplitude in FI_data[cell].keys():
            if current_amplitude not in average_firing_rates:
                average_firing_rates[current_amplitude] = []
                std_error_firing_rates[current_amplitude] = []
            average_firing_rates[current_amplitude].append(FI_data[cell][current_amplitude])
            std_error_firing_rates[current_amplitude].append(FI_data[cell][current_amplitude])

    for current_amplitude in average_firing_rates.keys():
        average_firing_rates[current_amplitude] = np.mean(average_firing_rates[current_amplitude])
        std_error_firing_rates[current_amplitude] = np.std(std_error_firing_rates[current_amplitude])/ np.sqrt(len(std_error_firing_rates[current_amplitude]))

    #sort the average firing rates by the current amplitudes
    average_firing_rates = dict(sorted(average_firing_rates.items()))

    #create a list of the current amplitudes and firing rates
    current_amplitudes = []
    firing_rates = []
    std_error = []

    #only plot values up to 400 and values from 0 to 400 by 50
    for key in average_firing_rates.keys():
        if key <= 400 and key >= 0 and key % 50 == 0:
            current_amplitudes.append(key)
            firing_rates.append(average_firing_rates[key])
            std_error.append(std_error_firing_rates[key])
    

    #convert the F_I data for each cell to a .csv file then create another csv file for the average firing rates, current amplitudes and standard error
    FI_data_df_all = pd.DataFrame(FI_data)
    FI_data_df_all.to_csv(f'{saved_path}_FI_data.csv')

    FI_summary_df = pd.DataFrame({'current_amplitudes': current_amplitudes, 'firing_rates': firing_rates, 'std_error': std_error})
    FI_summary_df.to_csv(f'{saved_path}_FI_summary.csv')

    #plot the current amplitudes and firing rates
    fig, ax = plt.subplots()

    ax.errorbar(current_amplitudes, firing_rates, yerr=std_error, fmt='o', color='black', ecolor='red', capsize=5)
    ax.set_xlabel('Current Amplitude (pA)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'{plot_title} N={len(FI_data)}')
    plt.show()

    return firing_rates, current_amplitudes, std_error

#plot the F_I curve for each cell and the input resistance over time for each cell
def get_input_resistances_multiple_cells(dir_path, current_pulse_amp, start_time, end_time):

    input_resistances_to_plot = {}
    time_points_to_plot = {}

    pkl_files = []
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.pkl'):
                pkl_files.append(os.path.join(path, name))
                cell_name = name.split('_')[0]
                cell_number = name.split('_')[1]
                full_cell_name = cell_name + '_' + cell_number
                input_resistances_to_plot[full_cell_name] = {}

                #open all of the pickle files and create a dataframe of their analysis_dicts
                for file in pkl_files:
                    data_file = pd.read_pickle(file)
                    acquisition_frequency = data_file['acquisition_frequency'][0]
                    current_pulse_amp = current_pulse_amp
                    start_time = start_time
                    end_time = end_time

                    input_resistances = []

                    all_sweeps = data_file['sweep'][:]
                    #Get the array from all the sweeps and store them in a list of traces
                    traces = []
                    for sweep in all_sweeps:
                        traces.append(sweep)

                    for trace in traces:
                        input_resistance = calculate_input_resistance(trace, current_pulse_amp, acquisition_frequency, start_time, end_time)
                        input_resistances.append(input_resistance)

                    input_resistances_to_plot[full_cell_name] = input_resistances

                    #create the time array
                    time_points = []
                    #enumerate through all the sweeps
                    for i in range(len(data_file)):
                        first_time = data_file['global_wall_clock'][0]
                        time_point = (data_file['global_wall_clock'][i] - first_time)/60
                        time_points.append(time_point)

                    time_points_to_plot[full_cell_name] = time_points

    return input_resistances_to_plot, time_points_to_plot


def plot_F_I_data_single_cell(single_cell_FI_data, ax, plot_title):
        for current_amplitude in single_cell_FI_data:
            current_amplitudes = []
            firing_rates = []
            for key, value in single_cell_FI_data.items():
                current_amplitudes.append(key)
                firing_rates.append(value)
            ax.scatter(current_amplitudes, firing_rates, color='black')
            ax.set_title(plot_title)
            ax.set_xlabel('Current Amplitude (pA)')
            ax.set_ylabel('Firing Rate (Hz)')

# single_cell = NMDG_FI_data['05162024_c3']

# fig, ax = plt.subplots()
# plot_F_I_data_single_cell(single_cell, ax, 'NMDG')
import os
import pandas as pd

def get_FI_traces_multiple_cells(dir_path):
# Initialize a dictionary to store traces by cell name
    pkl_files_traces = {}

    dir_path = dir_path

    # Traverse through the directory to collect .pkl files
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.pkl'):
                cell_name = name.split('_')[0]
                cell_number = name.split('_')[1]
                full_cell_name = cell_name + '_' + cell_number

                # Initialize the cell entry if it does not exist
                if full_cell_name not in pkl_files_traces:
                    pkl_files_traces[full_cell_name] = {}

                file_path = os.path.join(path, name)
                data_file = pd.read_pickle(file_path)

                # Collect current amplitudes and traces
                for i in range(len(data_file)):
                    analysis_data = data_file['analysis_dict'][i]

                    if 'Coarse_FI' in analysis_data.keys():
                        trace = data_file['sweep'][i]
                        current_amplitude = analysis_data['Coarse_FI']['current_amplitudes']
                    
                    elif 'IV_stim' in analysis_data.keys():
                        trace = data_file['sweep'][i]
                        current_amplitude = analysis_data['IV_stim']['current_amplitudes']

                        # Ensure current_amplitude is an integer and non nan
                        if not pd.isna(current_amplitude):
                            current_amplitude = int(current_amplitude)

                        # Initialize the list if it does not exist
                        if current_amplitude not in pkl_files_traces[full_cell_name]:
                            pkl_files_traces[full_cell_name][current_amplitude] = []

                        # Append the trace to the corresponding current amplitude
                        pkl_files_traces[full_cell_name][current_amplitude].append(trace)
            
    return pkl_files_traces


def plot_comparison(ax, x1, x2, group1_data, group2_data, y_pos, offset, p_offset, color='black'):
    valid_group1 = [val for val in group1_data if not np.isnan(val) and val > 0]
    valid_group2 = [val for val in group2_data if not np.isnan(val) and val > 0]
    
    if valid_group1 and valid_group2:
        t_stat, p_val = stats.ttest_ind(valid_group1, valid_group2, equal_var=False)
        ax.text((x1 + x2) / 2, y_pos + p_offset + 1, f'p = {np.round(p_val, 4)}', horizontalalignment='center', verticalalignment='center', fontsize=10)
        ax.plot([x1, x2], [y_pos + offset, y_pos + offset], color=color, linestyle='-', linewidth=1.0)

# Plot mean and error bars
def plot_means_with_error(ax, x_pos, data):
    mean_val = np.nanmean(data)
    std_error = np.nanstd(data) / np.sqrt(len(data))
    ax.bar(x_pos, mean_val, yerr=std_error, color='grey', alpha=0.5)
    ax.errorbar(x_pos, mean_val, yerr=std_error, fmt='o', color='black')
    return mean_val, std_error

def plot_intrinsic_properties_multiple_cells(intrinsic_properties_dict, ax):
    count = 0

    all_input_resistances = []
    all_voltage_sags = []
    for cell in intrinsic_properties_dict:
        if not intrinsic_properties_dict[cell]:
            continue
        all_input_resistances.append(intrinsic_properties_dict[cell]['steady_state_input_resistance'])
        all_voltage_sags.append(intrinsic_properties_dict[cell]['Voltage_sag'])
    

    for cell in intrinsic_properties_dict:
        if not intrinsic_properties_dict[cell]:
            continue
        ax[0].scatter(count, intrinsic_properties_dict[cell]['steady_state_input_resistance'], color='black', s = 50)
        ax[1].scatter(count, intrinsic_properties_dict[cell]['Voltage_sag'], color='black', s = 50)

    ax[0].set_xticks([])
    ax[0].set_title('Input Resistance')
    ax[0].set_ylabel('Input Resistance (MOhm)')

    ax[1].set_xticks([])
    ax[1].set_title('Voltage Sag')
    ax[1].set_ylabel('Voltage Sag (mV)')

    avg_input_resistance = np.mean(all_input_resistances)
    avg_voltage_sag = np.mean(all_voltage_sags)

    std_error_input_resistance = np.std(all_input_resistances)/np.sqrt(len(all_input_resistances))
    std_error_voltage_sag = np.std(all_voltage_sags)/np.sqrt(len(all_voltage_sags))

    ax[0].errorbar(count, avg_input_resistance, yerr=std_error_input_resistance, color='black', fmt='o')
    ax[1].errorbar(count, avg_voltage_sag, yerr=std_error_voltage_sag, color='black', fmt='o')

    ax[0].scatter(count, avg_input_resistance, color='black', s = 50)
    ax[1].scatter(count, avg_voltage_sag, color='black', s = 50)
    ax[0].bar(count, avg_input_resistance, yerr=std_error_input_resistance, color='grey', alpha=0.5)
    ax[1].bar(count, avg_voltage_sag, yerr=std_error_voltage_sag, color='grey', alpha=0.5)

    return all_input_resistances, all_voltage_sags

def get_ISI_times(dir_path):
    ISI_data = {}
    excluded_cells = []

    # Iterate through all files in the directory
    for path, subdirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith('.pkl'):
                cell_name = name.split('_')[0]
                cell_number = name.split('_')[1]
                full_cell_name = f"{cell_name}_{cell_number}"

                if full_cell_name not in ISI_data:
                    ISI_data[full_cell_name] = {}  # Initialize as a dictionary

                file_path = os.path.join(path, name)
                data_file = pd.read_pickle(file_path)
                analysis_data = data_file.get('analysis_dict', [])

                cell_firing_rates_above_20 = False

                for analysis in analysis_data:
                    if 'Coarse_FI' in analysis.keys() and 'AP' in analysis.keys():
                        current_amplitudes = np.unique(np.round(analysis['Coarse_FI']['current_amplitudes'], 1))
                        firing_rates = analysis['Coarse_FI']['firing_rates']
                        ISI_times = analysis['AP']['AP_ISI_time']
                    elif 'IV_stim' in analysis.keys() and 'AP' in analysis.keys():
                        current_amplitudes = np.unique(np.round(analysis['IV_stim']['current_amplitudes'], 1))
                        firing_rates = analysis['IV_stim']['firing_rates']
                        ISI_times = analysis['AP']['AP_ISI_time']
                    else:
                        continue

                    # Check if any firing rates exceed 20
                    if isinstance(firing_rates, (list, np.ndarray)):
                        if any(rate > 20 for rate in firing_rates):
                            cell_firing_rates_above_20 = True
                    else:
                        if firing_rates > 20:
                            cell_firing_rates_above_20 = True

                    # Record ISI times
                    for amplitude in current_amplitudes:
                        if amplitude not in ISI_data[full_cell_name]:
                            ISI_data[full_cell_name][amplitude] = {}
                        
                        for AP_number, ISI_time in enumerate(ISI_times):
                            AP_key = AP_number + 1  # Start AP number from 1
                            
                            # Only record ISI times where AP_number >= 2
                            if AP_key >= 2:
                                if AP_key not in ISI_data[full_cell_name][amplitude]:
                                    ISI_data[full_cell_name][amplitude][AP_key] = []

                                ISI_data[full_cell_name][amplitude][AP_key].append(ISI_time)

                # Average ISI times after collection
                for amplitude in ISI_data[full_cell_name]:
                    for AP_key in ISI_data[full_cell_name][amplitude]:
                        ISI_times_list = ISI_data[full_cell_name][amplitude][AP_key]
                        ISI_data[full_cell_name][amplitude][AP_key] = np.nanmean(ISI_times_list)

                # If no firing rates above 20, add to excluded cells list
                if not cell_firing_rates_above_20:
                    excluded_cells.append(name)
    
    print("Cells with no firing rates above 20:")
    for cell in excluded_cells:
        print(cell)

    return ISI_data, excluded_cells 

def get_ISI_times_slopes(ISI_data): 
    ISI_slopes = {}

    for cell in ISI_data: 
        sorted_ISI_times = []
        
        # Combine all ISI times across all current amplitudes
        for current_amplitude_data in ISI_data[cell]: 
            sorted_AP_numbers = sorted(ISI_data[cell][current_amplitude_data].keys())
            sorted_ISI_times.extend([ISI_data[cell][current_amplitude_data][AP_number] 
                                     for AP_number in sorted_AP_numbers])
        
        # Sort the combined ISI times
        sorted_ISI_times = sorted(sorted_ISI_times)
        
        # Calculate slope if there are at least 7 ISI times
        if len(sorted_ISI_times) >= 7:
            slope = np.nanmean(np.diff(sorted_ISI_times[:7]))  # Calculate the slope for the first 7 ISI times
            ISI_slopes[cell] = slope

    return ISI_slopes

def get_FI_rheobase_traces(data_dir, sweep_list, AP_trace_end):
    # Initialize a dictionary to store traces by cell name
    rheobase_traces = {}

    sweep_list = sweep_list

    data_files = os.listdir(data_dir)
    data_files = [file for file in data_files if file.endswith('.pkl')]

    sorted_data_files = []
    for data_file in data_files:
        # Read in the data file
        data_df = pd.read_pickle(data_dir + '/' + data_file)
        # Get the name of the cell - everything in the file name except '_processed_new.pkl'
        cell_date = data_file.split('_')[0]
        cell_name = data_file.split('_')[1]
        data_name = cell_name + '_' + cell_date
        # Find the date pattern of the data_name and sort the data by date
        date = data_name.split('_')[1]
        sorted_data_files.append((date, data_name, data_df))
        
    sorted_data_files.sort()

    for idx, sweep in enumerate(sweep_list):
        current_analysis_data = sorted_data_files[idx][2]['analysis_dict'][sweep]
        AP_threshold_indices = current_analysis_data['AP']['AP_threshold_indices']

        if len(AP_threshold_indices) > 1:
            #get the first AP threshold index
            AP_threshold_indices = int(AP_threshold_indices[0])
        
        #check if the AP threshold index and AHP index are lists
        if isinstance(AP_threshold_indices, list):
            AP_threshold_indices = AP_threshold_indices[0]

        data_name = sorted_data_files[idx][1]
        current_rheobase_trace = sorted_data_files[idx][2]['sweep'][sweep]

        #index the trace from the AP threshold to the  AHP index
        current_rheobase_trace = current_rheobase_trace[AP_threshold_indices: AP_threshold_indices +  int(AP_trace_end * 20000 / 1000)]

        if data_name not in rheobase_traces:
            rheobase_traces[data_name] = {}
        
        if 'Rheobase_Trace' not in rheobase_traces[data_name]:
            rheobase_traces[data_name]['Rheobase_Trace'] = []

        rheobase_traces[data_name]['Rheobase_Trace'].append(current_rheobase_trace)

        if 'Fine_FI' in current_analysis_data or 'IV_stim' in current_analysis_data or 'Coarse_FI' in current_analysis_data:
            if 'Rheobase_Current' not in rheobase_traces[data_name]:
                rheobase_traces[data_name]['Rheobase_Current'] = []
            
            if 'Fine_FI' in current_analysis_data and 'current_amplitudes' in current_analysis_data['Fine_FI']:
                rheobase_traces[data_name]['Rheobase_Current'].append(current_analysis_data['Fine_FI']['current_amplitudes'])
            elif 'IV_stim' in current_analysis_data and 'current_amplitudes' in current_analysis_data['IV_stim']:
                rheobase_traces[data_name]['Rheobase_Current'].append(current_analysis_data['IV_stim']['current_amplitudes'])
            elif 'Coarse_FI' in current_analysis_data and 'current_amplitudes' in current_analysis_data['Coarse_FI']:
                rheobase_traces[data_name]['Rheobase_Current'].append(current_analysis_data['Coarse_FI']['current_amplitudes'])
        

    return rheobase_traces


'''Functions for plotting E/I data'''

def get_E_I_traces(data_dir, unitary_stim_starts):
    data_files = [file for file in os.listdir(data_dir) if file.endswith('.pkl')]

    E_I_data_traces = {}
    for data_file in data_files:
        data_file_name = os.path.splitext(data_file)[0]
        current_data_df = pd.read_pickle(os.path.join(data_dir, data_file))

        for i in range(len(current_data_df)):
            stimulus_metadata = current_data_df['stimulus_metadata_dict'].iloc[i]
            if stimulus_metadata and 'ISI' in stimulus_metadata:
                ISI_value = stimulus_metadata['ISI']
                if ISI_value and ISI_value != 'nan' and not pd.isna(ISI_value):
                    ISI_time = int(float(ISI_value))  # Convert ISI to integer
                    condition = stimulus_metadata['condition']
                    
                    # Initialize E_I_data_traces for this file, ISI_time, and channels
                    if data_file_name not in E_I_data_traces:
                        E_I_data_traces[data_file_name] = {}
                    if ISI_time not in E_I_data_traces[data_file_name]:
                        E_I_data_traces[data_file_name][ISI_time] = {}

                    entry = current_data_df.iloc[i]
                    if 'E_I_pulse' in entry['analysis_dict']:
                        channels = entry['analysis_dict']['E_I_pulse'].keys()

                        for channel in channels:
                            if channel not in E_I_data_traces[data_file_name][ISI_time]:
                                E_I_data_traces[data_file_name][ISI_time][channel] = {}

                            if condition not in E_I_data_traces[data_file_name][ISI_time][channel]:
                                trace_dict = {
                                    'unitary_average_traces': [] if ISI_time == 300 else None,
                                    'unitary_all_traces': [] if ISI_time == 300 else None,
                                    'non_unitary_average_traces': None if ISI_time == 300 else [],
                                    'non_unitary_all_traces': None if ISI_time == 300 else [],
                                    'holding_potential': []
                                }
                                E_I_data_traces[data_file_name][ISI_time][channel][condition] = trace_dict
                            
                            partitioned_traces = entry['intermediate_traces'].get('partitioned_trace', {})
                            offset_trace = entry['intermediate_traces'].get('offset_trace', {})

                            if offset_trace:
                                holding_potentials = []
                                all_traces = []

                                if ISI_time == 300:  # Handle unitary traces for ISI 300
                                    for stim_start in offset_trace.get(channel, {}):
                                        if stim_start in unitary_stim_starts.get(channel, []):
                                            trace = offset_trace[channel][stim_start][0]
                                            if isinstance(trace, np.ndarray) and trace.size > 0:
                                                all_traces.append(trace)

                                    for stim_start in partitioned_traces.get(channel, {}):
                                        if stim_start in unitary_stim_starts.get(channel, []):
                                            trace = partitioned_traces[channel][stim_start][0]
                                            if isinstance(trace, np.ndarray) and trace.size > 0:
                                                baseline_voltage = np.mean(trace[:int(10 * 20000 / 1000)])
                                                holding_potentials.append(baseline_voltage)

                                    if all_traces:
                                        min_length = min(trace.shape[0] for trace in all_traces)
                                        all_traces = [trace[:min_length] for trace in all_traces]
                                        average_baseline = np.mean(holding_potentials)
                                        
                                        E_I_data_traces[data_file_name][ISI_time][channel][condition]['unitary_all_traces'].extend(all_traces)
                                        E_I_data_traces[data_file_name][ISI_time][channel][condition]['holding_potential'].append(average_baseline)
                                        unitary_average = np.mean(all_traces, axis=0)
                                        E_I_data_traces[data_file_name][ISI_time][channel][condition]['unitary_average_traces'] = unitary_average

                                else:  # Handle non-unitary traces for non-300 ISI
                                    if channel in offset_trace:
                                        trace = offset_trace[channel]
                                        if isinstance(trace, np.ndarray) and trace.size > 0:
                                            all_traces.append(trace)

                                    if channel in partitioned_traces:
                                        trace = partitioned_traces[channel]
                                        if isinstance(trace, np.ndarray) and trace.size > 0:
                                            baseline_voltage = np.mean(trace[:int(10 * 20000 / 1000)])
                                            holding_potentials.append(baseline_voltage)

                                    if all_traces:
                                        min_length = min(trace.shape[0] for trace in all_traces)
                                        all_traces = [trace[:min_length] for trace in all_traces]
                                        average_baseline = np.mean(holding_potentials)

                                        E_I_data_traces[data_file_name][ISI_time][channel][condition]['non_unitary_all_traces'].extend(all_traces)
                                        E_I_data_traces[data_file_name][ISI_time][channel][condition]['holding_potential'].append(average_baseline)
                                        non_unitary_average = np.mean(all_traces, axis=0)
                                        E_I_data_traces[data_file_name][ISI_time][channel][condition]['non_unitary_average_traces'] = non_unitary_average

    return E_I_data_traces

def extract_holding_potentials(E_I_traces_dict):
    holding_potentials_dict = {}
    data = E_I_traces_dict 
    for cell in data:
        if cell not in holding_potentials_dict:
            holding_potentials_dict[cell] = {}
        for ISI_time in data[cell]:
            if ISI_time not in holding_potentials_dict[cell]:
                holding_potentials_dict[cell][ISI_time] = {}
            for channel in data[cell][ISI_time]:
                if channel not in holding_potentials_dict[cell][ISI_time]:
                    holding_potentials_dict[cell][ISI_time][channel] = {}
                for condition in data[cell][ISI_time][channel]:
                    if condition not in holding_potentials_dict[cell][ISI_time][channel]:
                        holding_potentials_dict[cell][ISI_time][channel][condition] = {'holding_potential': []} 

                        holding_potentials = data[cell][ISI_time][channel][condition]['holding_potential']
                        holding_potential_average = np.mean(holding_potentials)
                        holding_potentials_dict[cell][ISI_time][channel][condition]['holding_potential'].append(holding_potential_average)
    
    return holding_potentials_dict

def plot_all_traces_E_I_example(current_experiment, ax):
    for channel in current_experiment:
        for condition in current_experiment[channel]:
            if condition == 'Control':
                if 'unitary_all_traces' in current_experiment[channel][condition]:
                    if channel == 'channel_1':
                        for trace in current_experiment[channel][condition]['unitary_all_traces']:
                            current_time = np.arange(0, len(trace) * 0.05, 0.05)
                            ax[0].plot(current_time, trace, color='black', alpha=0.8)
                    if channel == 'channel_2':
                        for trace in current_experiment[channel][condition]['unitary_all_traces']:
                            current_time = np.arange(0, len(trace) * 0.05, 0.05)
                            ax[1].plot(current_time, trace, color='black', alpha=0.8)
                if 'non_unitary_all_traces' in current_experiment[channel][condition]:
                    if channel == 'channel_1':
                        for trace in current_experiment[channel][condition]['non_unitary_all_traces']:
                            current_time = np.arange(0, len(trace) * 0.05, 0.05)
                            ax[0].plot(current_time, trace, color='black', alpha=0.8)
                    if channel == 'channel_2':
                        for trace in current_experiment[channel][condition]['non_unitary_all_traces']:
                            current_time = np.arange(0, len(trace) * 0.05, 0.05)
                            ax[1].plot(current_time, trace, color='black', alpha=0.8)


            if condition == 'Gabazine':
                if 'unitary_all_traces' in current_experiment[channel][condition]:
                    if channel == 'channel_1':
                        for trace in current_experiment[channel][condition]['unitary_all_traces']:
                            current_time = np.arange(0, len(trace) * 0.05, 0.05)
                            ax[0].plot(current_time, trace, color='red', alpha=0.8)
                    if channel == 'channel_2':
                        for trace in current_experiment[channel][condition]['unitary_all_traces']:
                            current_time = np.arange(0, len(trace) * 0.05, 0.05)
                            ax[1].plot(current_time, trace, color='red', alpha=0.8)
                if 'non_unitary_all_traces' in current_experiment[channel][condition]:
                    if channel == 'channel_1':
                        for trace in current_experiment[channel][condition]['non_unitary_all_traces']:
                            current_time = np.arange(0, len(trace) * 0.05, 0.05)
                            ax[0].plot(current_time, trace, color='red', alpha=0.8)

def plot_average_trace_example(current_experiment, ax):
    #plot the average traces
    for channel in current_experiment:
        for condition in current_experiment[channel]:
            if condition == 'Control':
                if 'unitary_average_traces' in current_experiment[channel][condition]:
                    if channel == 'channel_1':
                        current_time = np.arange(0, len(current_experiment[channel][condition]['unitary_average_traces']) * 0.05, 0.05)
                        ax[0].plot(current_time, current_experiment[channel][condition]['unitary_average_traces'], color='black', linewidth=2)
                    if channel == 'channel_2':
                        current_time = np.arange(0, len(current_experiment[channel][condition]['unitary_average_traces']) * 0.05, 0.05)
                        ax[1].plot(current_time, current_experiment[channel][condition]['unitary_average_traces'], color='black', linewidth=2)
                if 'non_unitary_average_traces' in current_experiment[channel][condition]:
                    if channel == 'channel_1':
                        current_time = np.arange(0, len(current_experiment[channel][condition]['non_unitary_average_traces']) * 0.05, 0.05)
                        ax[0].plot(current_time, current_experiment[channel][condition]['non_unitary_average_traces'], color='black', linewidth=2)
                    if channel == 'channel_2':
                        current_time = np.arange(0, len(current_experiment[channel][condition]['non_unitary_average_traces']) * 0.05, 0.05)
                        ax[1].plot(current_time, current_experiment[channel][condition]['non_unitary_average_traces'], color='black', linewidth=2)

            if condition == 'Gabazine':
                if 'unitary_average_traces' in current_experiment[channel][condition]:
                    if channel == 'channel_1':
                        current_time = np.arange(0, len(current_experiment[channel][condition]['unitary_average_traces']) * 0.05, 0.05)
                        ax[0].plot(current_time, current_experiment[channel][condition]['unitary_average_traces'], color='red', linewidth=2)
                    if channel == 'channel_2':
                        current_time = np.arange(0, len(current_experiment[channel][condition]['unitary_average_traces']) * 0.05, 0.05)
                        ax[1].plot(current_time, current_experiment[channel][condition]['unitary_average_traces'], color='red', linewidth=2)
                if 'non_unitary_average_traces' in current_experiment[channel][condition]:
                    if channel == 'channel_1':
                        current_time = np.arange(0, len(current_experiment[channel][condition]['non_unitary_average_traces']) * 0.05, 0.05)
                        ax[0].plot(current_time, current_experiment[channel][condition]['non_unitary_average_traces'], color='red', linewidth=2)
                    if channel == 'channel_2':
                        current_time = np.arange(0, len(current_experiment[channel][condition]['non_unitary_average_traces']) * 0.05, 0.05)
                        ax[1].plot(current_time, current_experiment[channel][condition]['non_unitary_average_traces'], color='red', linewidth=2)

def create_expected_EPSP(unitary_EPSP, stim_times):
    baseline_len = int(10 * 20000 / 1000)
    unitary_len = len(unitary_EPSP)
    total_len = int(800 * 20000 / 1000)
    compound_EPSP = np.zeros(total_len)

    for stim in stim_times:
        stim_index = int(stim * 20000 / 1000)
        start = baseline_len + stim_index

        # Ensure we do not go out of bounds
        end = start + unitary_len
        if start < total_len:
            if end <= total_len:
                compound_EPSP[start:end] += unitary_EPSP
            else:
                # Handle the case where the unitary_EPSP would exceed the bounds of compound_EPSP
                overlap_len = total_len - start
                if overlap_len > 0:
                    compound_EPSP[start:] += unitary_EPSP[:overlap_len]

    return compound_EPSP 

def get_compound_EPSPs(experiment_dict, ISI_times_dict):
    try:
        unitary_EPSPs = {}
        for channel in experiment_dict:
            unitary_EPSPs[channel] = {}
            for condition in experiment_dict[channel]:
                unitary_EPSPs[channel][condition] = {}
                if 'unitary_average_traces' in experiment_dict[channel][condition] and experiment_dict[channel][condition]['unitary_average_traces'].any():
                    unitary_EPSPs[channel][condition] = experiment_dict[channel][condition]['unitary_average_traces']

        compound_EPSP_ISI = {}
        for ISI in ISI_times_dict:
            compound_EPSP_ISI[ISI] = {}
            for channel in unitary_EPSPs:
                if 'Gabazine' or 'gabazine' in unitary_EPSPs[channel]:
                    compound_EPSP_ISI[ISI][channel] = create_expected_EPSP(unitary_EPSPs[channel]['Gabazine'], ISI_times_dict_old[ISI])
        
        return compound_EPSP_ISI
    
    except Exception as e:
        print(f"Error generating compound EPSPs: {e}")
        return None
    
def plot_all_ISI_traces_example(compound_EPSPs, non_unitary_EPSPs, baseline_duration, buffer_time, title):
    acquisition_frequency = 20000
    baseline_duration = baseline_duration # in ms
    baseline_index = int(baseline_duration * acquisition_frequency / 1000)  # Convert baseline duration to index
    buffer_index = int(buffer_time * acquisition_frequency / 1000)  # Convert 10 ms buffer to index


    fig, ax = plt.subplots(1, 1, figsize=(6, 5)) 
    #add a super title to the top and make it fit the figure
    fig.suptitle(title, fontsize=16)

    # Plot each ISI time for non-unitary EPSPs by subtracting the 10 ms buffer
    for i, ISI_time in enumerate(non_unitary_EPSPs):
        for channel in non_unitary_EPSPs[ISI_time]:
            if channel == 'channel_1':
                current_experiment = non_unitary_EPSPs[ISI_time][channel]
                for color, condition in zip(['black', 'red'], current_experiment):
                    current_trace = current_experiment[condition][buffer_index:]  # Subtract 10 ms buffer
                    current_time = np.arange(0, len(current_trace) * 0.05, 0.05)  # Start from 0 ms
                    
                    ax[i+1, 0].plot(current_time, current_trace, label=f'{condition}', alpha=0.5, color = color)
                    ax[i+1, 0].set_title(f'{ISI_time} Perforant Pathway')
                    ax[i+1, 0].set_xlabel('Time (ms)')
                    ax[i+1, 0].set_ylabel('EPSP Amplitude (mV)')
                    ax[i+1, 0].legend()
                    
            if channel == 'channel_2':
                current_experiment = non_unitary_EPSPs[ISI_time][channel]
                for color, condition in zip(['black', 'red'], current_experiment):
                    current_trace = current_experiment[condition][buffer_index:]  # Subtract 10 ms buffer
                    current_time = np.arange(0, len(current_trace) * 0.05, 0.05)  # Start from 0 ms
                    
                    ax[i+1, 1].plot(current_time, current_trace, label=f'{condition}', alpha=0.5, color = color)
                    ax[i+1, 1].set_title(f'{ISI_time} Schaffer Collateral Pathway')
                    ax[i+1, 1].set_xlabel('Time (ms)')
                    ax[i+1, 1].set_ylabel('EPSP Amplitude (mV)')
                    ax[i+1, 1].legend()

    # Plot the expected EPSPs for each ISI by subtracting the 500 ms baseline from the compound traces
    for i, ISI in enumerate(compound_EPSPs):
        for channel in compound_EPSPs[ISI]:
            if channel == 'channel_1':
                current_compound_trace_perforant = compound_EPSPs[ISI][channel][baseline_index:]  # Subtract baseline
                # updated_perforant_trace = np.where(current_compound_trace_perforant < 0, 0, current_compound_trace_perforant)
                current_time = np.arange(0, len(current_compound_trace_perforant) * 0.05, 0.05)
                
                ax[i, 0].plot(current_time, current_compound_trace_perforant, label=f'Expected - Linear Summation', color='#ADD8E6')
                ax[i, 0].set_title(f'{ISI} Perforant Pathway')
                ax[i, 0].set_xlabel('Time (ms)')
                ax[i, 0].set_ylabel('EPSP Amplitude (mV)')
                ax[i, 0].legend()

            if channel == 'channel_2':
                current_compound_trace_schaffer = compound_EPSPs[ISI][channel][baseline_index:]  # Subtract baseline
                # updated_schaffer_trace = np.where(current_compound_trace_schaffer < 0, 0, current_compound_trace_schaffer)
                current_time = np.arange(0, len(current_compound_trace_schaffer) * 0.05, 0.05)

                ax[i, 1].plot(current_time, current_compound_trace_schaffer, label=f'Expected - Linear Summation', color='#ADD8E6')
                ax[i, 1].set_title(f'{ISI} Schaffer Collateral Pathway')
                ax[i, 1].set_xlabel('Time (ms)')
                ax[i, 1].set_ylabel('EPSP Amplitude (mV)')
                ax[i, 1].legend()
#now with traces in hand get the peaks of the average traces
#get the peaks of the average traces for all ISI experiments
#get the EPSP amplitudes from peaks of averaged traces
#from the amplitudes then calculate the estimated inhibition (Gabazine - Control)
#from the estimated Inhibition calculate the E/I imbalance (Gabazine)/(Gabazine + estimated inhibition)


def get_E_I_amplitudes_and_estimated_inhibition_traces(E_I_traces_dict):
    E_I_data_amplitudes = {}
    for cell in E_I_traces_dict:
        if cell not in E_I_data_amplitudes:
            E_I_data_amplitudes[cell] = {}

        for ISI in E_I_traces_dict[cell]:
            if ISI not in E_I_data_amplitudes[cell]:
                E_I_data_amplitudes[cell][ISI] = {}

            for channel in E_I_traces_dict[cell][ISI]:
                if channel not in E_I_data_amplitudes[cell][ISI]:
                    E_I_data_amplitudes[cell][ISI][channel] = {}

                for condition in E_I_traces_dict[cell][ISI][channel]:
                    #print(condition)
                    if condition not in E_I_data_amplitudes[cell][ISI][channel]:
                        E_I_data_amplitudes[cell][ISI][channel][condition] = {}
                    if ISI == 300:
                        if 'unitary_average_traces' in E_I_traces_dict[cell][ISI][channel][condition]:
                            current_trace = E_I_traces_dict[cell][ISI][channel][condition]['unitary_average_traces']
                            working_trace = current_trace.copy()

                            # Find peaks in the working trace
                            peaks, _ = find_peaks(working_trace, height=1)
                            if peaks.size > 0:
                                # Get the max peak value and its index
                                max_peak = max(working_trace[peaks])
                                max_peak_index = np.where(working_trace == max_peak)[0][0]

                                # Append the peak value and index to the dictionary
                                E_I_data_amplitudes[cell][ISI][channel][condition]['max_peak_value'] = max_peak
                                E_I_data_amplitudes[cell][ISI][channel][condition]['max_peak_idx'] = max_peak_index
                    else:
                        if 'non_unitary_average_traces' in E_I_traces_dict[cell][ISI][channel][condition]:
                            current_trace = E_I_traces_dict[cell][ISI][channel][condition]['non_unitary_average_traces']
                            working_trace = current_trace.copy()

                            # Find peaks in the working trace
                            peaks, _ = find_peaks(working_trace, height=1)
                            if peaks.size > 0:
                                # Get the max peak value and its index
                                max_peak = max(working_trace[peaks])
                                max_peak_index = np.where(working_trace == max_peak)[0][0]

                                # Append the peak value and index to the dictionary
                                E_I_data_amplitudes[cell][ISI][channel][condition]['max_peak_value'] = max_peak
                                E_I_data_amplitudes[cell][ISI][channel][condition]['max_peak_idx'] = max_peak_index
                if 'Control' in E_I_traces_dict[cell][ISI][channel] and 'Gabazine' in E_I_traces_dict[cell][ISI][channel]:
                    working_trace = None
                    try:
                        if ISI == 300:
                            if 'unitary_average_traces' in E_I_traces_dict[cell][ISI][channel]['Control']:
                                E_I_traces_dict[cell][ISI][channel]['estimated_inhibition'] = {}
                                E_I_traces_dict[cell][ISI][channel]['estimated_inhibition']['unitary_average_traces'] = (
                                        E_I_traces_dict[cell][ISI][channel]['Gabazine']['unitary_average_traces'] -
                                        E_I_traces_dict[cell][ISI][channel]['Control']['unitary_average_traces'])
                                working_trace = E_I_traces_dict[cell][ISI][channel]['estimated_inhibition']['unitary_average_traces'].copy()

                        else:
                            if 'non_unitary_average_traces' in E_I_traces_dict[cell][ISI][channel]['Control']:
                                E_I_traces_dict[cell][ISI][channel]['estimated_inhibition'] = {}
                                E_I_traces_dict[cell][ISI][channel]['estimated_inhibition']['non_unitary_average_traces'] = (
                                        E_I_traces_dict[cell][ISI][channel]['Gabazine']['non_unitary_average_traces'] -
                                        E_I_traces_dict[cell][ISI][channel]['Control']['non_unitary_average_traces'])
                                working_trace = E_I_traces_dict[cell][ISI][channel]['estimated_inhibition'][
                                    'non_unitary_average_traces'].copy()
                        if 'estimated_inhibition' not in E_I_data_amplitudes[cell][ISI][channel]:
                            E_I_data_amplitudes[cell][ISI][channel]['estimated_inhibition'] = {}
                        condition = 'estimated_inhibition'

                        # Find peaks in the working trace
                        peaks, _ = find_peaks(working_trace, height=1)
                        if peaks.size > 0:
                            # Get the max peak value and its index
                            max_peak = max(working_trace[peaks])
                            max_peak_index = np.where(working_trace == max_peak)[0][0]

                            # Append the peak value and index to the dictionary
                            E_I_data_amplitudes[cell][ISI][channel][condition]['max_peak_value'] = max_peak
                            E_I_data_amplitudes[cell][ISI][channel][condition]['max_peak_idx'] = max_peak_index
                    except Exception as e:
                        print(
                            f"Error calculating estimated inhibition for cell {cell}, ISI {ISI}, channel {channel}: {e}")

    return E_I_data_amplitudes, E_I_traces_dict


def get_E_I_imbalance(E_I_amplitudes_dict):
    E_I_imbalances = {}

    for cell in E_I_amplitudes_dict:
        if cell not in E_I_imbalances:
            E_I_imbalances[cell] = {}

        for ISI in E_I_amplitudes_dict[cell]:
            if ISI not in E_I_imbalances[cell]:
                E_I_imbalances[cell][ISI] = {}

            for channel in E_I_amplitudes_dict[cell][ISI]:
                if channel not in E_I_imbalances[cell][ISI]:
                    E_I_imbalances[cell][ISI][channel] = []

                control_peak = None
                gabazine_peak = None
                estimated_inhibition_peak = None

                for condition, condition_data in E_I_amplitudes_dict[cell][ISI][channel].items():
                    if condition == 'Control':
                        control_peak = condition_data.get('max_peak_value', None)
                    elif condition == 'Gabazine':
                        gabazine_peak = condition_data.get('max_peak_value', None)
                    elif condition == 'estimated_inhibition':
                        estimated_inhibition_peak = condition_data.get('max_peak_value', None)

                # Iterate over the gabazine peaks
                if gabazine_peak is not None and estimated_inhibition_peak is not None:
                    if (gabazine_peak + estimated_inhibition_peak) != 0:  # Avoid division by zero
                        E_I_imbalance = gabazine_peak / (gabazine_peak + estimated_inhibition_peak)
                    else:
                        print('Warning: E_I_imbalance is zero for cell:', cell, 'ISI:', ISI, 'channel:', channel)
                        E_I_imbalance = 0  # Handle edge case where sum is zero
                    E_I_imbalances[cell][ISI][channel].append(E_I_imbalance)

    return E_I_imbalances


def get_E_I_imbalance_mean_std_error(E_I_imbalance_dict, ISI_times, channels): 
    E_I_means = {channel: {ISI_time: np.nan for ISI_time in ISI_times} for channel in channels}
    E_I_std_errors = {channel: {ISI_time: np.nan for ISI_time in ISI_times} for channel in channels} 

    all_data = {channel: {ISI_time: [] for ISI_time in ISI_times} for channel in channels}

    for cell in E_I_imbalance_dict:
        for ISI_time in ISI_times:
            if ISI_time in E_I_imbalance_dict[cell]:
                for channel in channels:
                    if channel in E_I_imbalance_dict[cell][ISI_time]:
                        current_data = E_I_imbalance_dict[cell][ISI_time][channel]
                        #convert the current data value to a float
                        current_data = [float(i) for i in current_data]
                        all_data[channel][ISI_time] += current_data

    for ISI_time in ISI_times: 
        for channel in channels:
            working_data = all_data[channel][ISI_time]
            #clean the data and make sure that there are no values greater than 1
            working_data = [i for i in working_data if i <= 1]
            E_I_means[channel][ISI_time] = np.nanmean(working_data)
            E_I_std_errors[channel][ISI_time] = np.nanstd(working_data) / np.sqrt(len(working_data))

    return E_I_means, E_I_std_errors 

def get_expected_EPSPs_multiple_cells(E_I_traces_dict, ISI_times_dict):
    #get compound EPSP then get the peaks of the compound EPSPs for all ISI experiments
    expected_EPSPs = {}
    expected_EPSPs_peaks = {}
    for cell in E_I_traces_dict:
        unitary_dict = E_I_traces_dict[cell][300]
        if cell not in expected_EPSPs:
            expected_EPSPs[cell] = {}
        if cell not in expected_EPSPs_peaks:
            expected_EPSPs_peaks[cell] = {}
        for channel in unitary_dict:
            if channel not in expected_EPSPs[cell]:
                expected_EPSPs[cell][channel] = {}
            if channel not in expected_EPSPs_peaks[cell]:
                expected_EPSPs_peaks[cell][channel] = {}
            if 'Gabazine' in unitary_dict[channel]:
                unitary_trace =  unitary_dict[channel]['Gabazine']['unitary_average_traces']

                #now create the compound EPSP for each ISI
                for ISI in ISI_times_dict:
                    if ISI not in expected_EPSPs[cell][channel]:
                        expected_EPSPs[cell][channel][ISI] = {}
                    if ISI not in expected_EPSPs_peaks[cell][channel]:
                        expected_EPSPs_peaks[cell][channel][ISI] = {}

                    expected_EPSP = create_expected_EPSP(unitary_trace, ISI_times_dict[ISI])
                    expected_EPSPs[cell][channel][ISI] = expected_EPSP

                    # Get the peaks of the compound EPSPs
                    peaks, _ = find_peaks(expected_EPSP, height=1)
                    #get the max peak value and its index
                    if peaks is not None and peaks.size > 0:
                        max_peak = max(expected_EPSP[peaks])
                        max_peak_index = np.where(expected_EPSP == max_peak)[0][0]
                        expected_EPSPs_peaks[cell][channel][ISI] = {
                            'max_peak_value': max_peak,
                            'max_peak_idx': max_peak_index
                            }


    return expected_EPSPs, expected_EPSPs_peaks

def plot_full_E_I_experiment(amplitude_dict, compound_EPSP_dict, ISI_times, channels, channel_names, conditions, colors, fix, ax):
    # Initialize dictionaries to hold the mean and standard error for each condition and ISI
    condition_data = {condition: {channel: np.full(len(ISI_times), np.nan) for channel in channels} for condition in conditions}
    std_err_data = {condition: {channel: np.full(len(ISI_times), np.nan) for channel in channels} for condition in conditions}

    # Iterate over each cell in E_I_amplitudes_NMDG
    for cell in amplitude_dict:
        for i, channel in enumerate(channels):
            for condition in conditions:
                for j, ISI_time in enumerate(ISI_times):
                    # Check if the ISI time exists in the dictionary
                    if ISI_time in amplitude_dict[cell]:
                        if channel in amplitude_dict[cell][ISI_time]:
                            if condition in amplitude_dict[cell][ISI_time][channel]:
                                if amplitude_dict[cell][ISI_time][channel][condition]['max_peak_value']:
                                    current_data = amplitude_dict[cell][ISI_time][channel][condition]['max_peak_value']
                                    all_data = current_data if j == 0 else np.concatenate((all_data, current_data))

                                if len(all_data) > 0:
                                    condition_data[condition][channel][j] = np.mean(all_data)
                                    std_err_data[condition][channel][j] = np.std(all_data) / np.sqrt(len(all_data))

    # Initialize dictionaries to hold the mean and standard error for each ISI and channel for compound EPSPs
    compound_EPSP_means = {channel: np.full(len(ISI_times), np.nan) for channel in channels}
    compound_EPSP_std_errors = {channel: np.full(len(ISI_times), np.nan) for channel in channels}
    compound_EPSP_data = {channel: {ISI_time: [] for ISI_time in ISI_times} for channel in channels}

    # Iterate over each cell in compound_EPSPs_peaks_NMDG
    for cell in compound_EPSP_dict:
        for i, channel in enumerate(channels):
            for j, ISI_time in enumerate(ISI_times):
                if channel in compound_EPSP_dict[cell]:
                    if ISI_time in compound_EPSP_dict[cell][channel]:
                        if 'max_peak_value' in compound_EPSP_dict[cell][channel][ISI_time]:
                            current_compound_data = compound_EPSP_dict[cell][channel][ISI_time]['max_peak_value']
                            compound_EPSP_data[channel][ISI_time].append(current_compound_data)

    # Calculate means and standard errors for compound EPSPs
    for channel in channels:
        for j, ISI_time in enumerate(ISI_times):
            if len(compound_EPSP_data[channel][ISI_time]) > 0:
                compound_EPSP_means[channel][j] = np.mean(compound_EPSP_data[channel][ISI_time])
                compound_EPSP_std_errors[channel][j] = np.std(compound_EPSP_data[channel][ISI_time]) / np.sqrt(len(compound_EPSP_data[channel][ISI_time]))

    # Plotting the data
    # Plotting the data
    for i, channel in enumerate(channels):
        for condition, color in zip(conditions, colors):
            ax[i].scatter(range(len(ISI_times)), condition_data[condition][channel], label=condition, color=color)
            ax[i].errorbar(range(len(ISI_times)), condition_data[condition][channel], yerr=std_err_data[condition][channel], fmt='o', color=color, capsize=5)
            ax[i].set_title(f'{channel_names[i]}')
            ax[i].set_xlabel('ISI Time (ms)')
            ax[i].set_ylabel('EPSP Amplitude (mV)')
            ax[i].set_xticks(range(len(ISI_times)))
            ax[i].set_xticklabels(ISI_times)
            ax[i].legend()
            
        # Plot compound EPSP peaks
        ax[i].scatter(range(len(ISI_times)), compound_EPSP_means[channel], label='Compound EPSP Peak', color='blue')
        ax[i].errorbar(range(len(ISI_times)), compound_EPSP_means[channel], yerr=compound_EPSP_std_errors[channel], fmt='o', color='blue', capsize=5)
        ax[i].set_title(f'{channel_names[i]} - Compound EPSP Peaks')
        ax[i].set_xlabel('ISI Time (ms)')
        ax[i].set_ylabel('Compound EPSP Peak (mV)')
        ax[i].set_xticks(range(len(ISI_times)))
        ax[i].set_xticklabels(ISI_times)
        ax[i].legend()

    plt.tight_layout()
    plt.show() 

def get_E_I_imbalance_mean_std_error(E_I_imbalance_dict, ISI_times, channels): 
    E_I_means = {channel: {ISI_time: np.nan for ISI_time in ISI_times} for channel in channels}
    E_I_std_errors = {channel: {ISI_time: np.nan for ISI_time in ISI_times} for channel in channels} 

    all_data = {channel: {ISI_time: [] for ISI_time in ISI_times} for channel in channels}

    for cell in E_I_imbalance_dict:
        for ISI_time in ISI_times:
            if ISI_time in E_I_imbalance_dict[cell]:
                for channel in channels:
                    if channel in E_I_imbalance_dict[cell][ISI_time]:
                        current_data = E_I_imbalance_dict[cell][ISI_time][channel]
                        #convert the current data value to a float
                        current_data = [float(i) for i in current_data]
                        all_data[channel][ISI_time] += current_data

    for ISI_time in ISI_times: 
        for channel in channels:
            working_data = all_data[channel][ISI_time]
            #clean the data and make sure that there are no values greater than 1
            working_data = [i for i in working_data if i <= 1]
            E_I_means[channel][ISI_time] = np.nanmean(working_data)
            E_I_std_errors[channel][ISI_time] = np.nanstd(working_data) / np.sqrt(len(working_data))

    return E_I_means, E_I_std_errors 

def plot_all_cells_graphs(E_I_traces, ISI_times_list, get_compound_EPSPs_func):
    for cell_name in E_I_traces:
        unitary_dict = E_I_traces[cell_name][300]
        non_unitary_data = E_I_traces[cell_name] 

        # Initialize the dictionary to store non-unitary EPSPs
        non_unitary_EPSPs = {}

        # Filter and extract relevant data based on ISI_list
        for ISI_time in ISI_times_list:
            non_unitary_EPSPs[ISI_time] = {}
            if ISI_time in non_unitary_data:
                for channel in non_unitary_data[ISI_time]:
                    non_unitary_EPSPs[ISI_time][channel] = {}
                    for condition in non_unitary_data[ISI_time][channel]:
                        if 'non_unitary_average_traces' in non_unitary_data[ISI_time][channel][condition]:
                            trace_data = non_unitary_data[ISI_time][channel][condition]['non_unitary_average_traces']
                            if trace_data.any():  # Ensure there is data to add
                                non_unitary_EPSPs[ISI_time][channel][condition] = trace_data

        compound_EPSPs = get_compound_EPSPs_func(unitary_dict, ISI_times_list)

        # Remove the 300 ISI time if present
        if 300 in non_unitary_EPSPs:
            del non_unitary_EPSPs[300]

        num_cells = len(E_I_traces)

        # Set up the figure and axes for plotting
        fig, ax = plt.subplots(5, 2, figsize=(15, 25))

        try:
            for ISI_time in compound_EPSPs:
                for channel in compound_EPSPs[ISI_time]:
                    if channel == 'channel_1':
                        current_compound_trace = compound_EPSPs[ISI_time][channel]
                        current_time = np.arange(0, len(current_compound_trace) * 0.05, 0.05)
                        ax[ISI_times_list.index(ISI_time), 0].plot(current_time, current_compound_trace, label=f'Expected - {ISI_time} ms ISI', alpha=0.5)
                        ax[ISI_times_list.index(ISI_time), 0].set_title(f'{ISI_time} Perforant Pathway - {cell_name}')
                        ax[ISI_times_list.index(ISI_time), 0].set_xlabel('Time (ms)')
                        ax[ISI_times_list.index(ISI_time), 0].set_ylabel('EPSP Amplitude (mV)')
                        ax[ISI_times_list.index(ISI_time), 0].legend()

                    if channel == 'channel_2':
                        current_compound_trace = compound_EPSPs[ISI_time][channel]
                        current_time = np.arange(0, len(current_compound_trace) * 0.05, 0.05)
                        ax[ISI_times_list.index(ISI_time), 1].plot(current_time, current_compound_trace, label=f'Expected - {ISI_time} ms ISI', alpha=0.5)
                        ax[ISI_times_list.index(ISI_time), 1].set_title(f'{ISI_time} Schaffer Collateral Pathway - {cell_name}')
                        ax[ISI_times_list.index(ISI_time), 1].set_xlabel('Time (ms)')
                        ax[ISI_times_list.index(ISI_time), 1].set_ylabel('EPSP Amplitude (mV)')
                        ax[ISI_times_list.index(ISI_time), 1].legend()
        except:
            pass

        for ISI_time in non_unitary_EPSPs:
            for channel in non_unitary_EPSPs[ISI_time]:
                if channel == 'channel_1':
                    current_experiment = non_unitary_EPSPs[ISI_time][channel]
                    for color, condition in zip(['black', 'red'], current_experiment):
                        current_trace = current_experiment[condition]
                        current_time = np.arange(0, len(current_trace) * 0.05, 0.05)
                        ax[ISI_times_list.index(ISI_time), 0].plot(current_time, current_trace, label=f'{condition}', alpha=0.5, color=color)
                        ax[ISI_times_list.index(ISI_time), 0].set_title(f'{ISI_time} Perforant Pathway - {cell_name}')
                        ax[ISI_times_list.index(ISI_time), 0].set_xlabel('Time (ms)')
                        ax[ISI_times_list.index(ISI_time), 0].set_ylabel('EPSP Amplitude (mV)')
                        ax[ISI_times_list.index(ISI_time), 0].legend()

                if channel == 'channel_2':
                    current_experiment = non_unitary_EPSPs[ISI_time][channel]
                    for color, condition in zip(['black', 'red'], current_experiment):
                        current_trace = current_experiment[condition]
                        current_time = np.arange(0, len(current_trace) * 0.05, 0.05)
                        ax[ISI_times_list.index(ISI_time), 1].plot(current_time, current_trace, label=f'{condition}', alpha=0.5, color=color)
                        ax[ISI_times_list.index(ISI_time), 1].set_title(f'{ISI_time} Schaffer Collateral Pathway - {cell_name}')
                        ax[ISI_times_list.index(ISI_time), 1].set_xlabel('Time (ms)')
                        ax[ISI_times_list.index(ISI_time), 1].set_ylabel('EPSP Amplitude (mV)')
                        ax[ISI_times_list.index(ISI_time), 1].legend()

        plt.tight_layout()
        plt.show()

def process_EPSP_data(combined_E_I_data):
    """
    Process EPSP data to extract control and gabazine data, and calculate mean and standard error.

    Parameters:
    - combined_WT_E_I_data: Dictionary containing EPSP amplitudes data for cells.

    Returns:
    - control_mean: Dictionary with structure {ISI_time: {channel: mean}} for Control condition.
    - control_std_error: Dictionary with structure {ISI_time: {channel: std_error}} for Control condition.
    - gabazine_mean: Dictionary with structure {ISI_time: {channel: mean}} for Gabazine condition.
    - gabazine_std_error: Dictionary with structure {ISI_time: {channel: std_error}} for Gabazine condition.
    """
    control_data = {} 
    gabazine_data = {}

    # Organize data by ISI time and channel
    for cell, cell_data in combined_E_I_data['EPSP_amplitudes'].items():
        for ISI_time, ISI_data in cell_data.items():
            control_data.setdefault(ISI_time, {}).setdefault('channel_1', [])
            control_data.setdefault(ISI_time, {}).setdefault('channel_2', [])
            gabazine_data.setdefault(ISI_time, {}).setdefault('channel_1', [])
            gabazine_data.setdefault(ISI_time, {}).setdefault('channel_2', [])
            
            for channel, channel_data in ISI_data.items():
                for condition, condition_data in channel_data.items():
                    max_peak_value = condition_data.get('max_peak_value', None)
                    if max_peak_value is not None:  # Ensure the value is not empty
                        if condition == 'Control':
                            control_data[ISI_time][channel].append(max_peak_value)
                        elif condition == 'Gabazine':
                            gabazine_data[ISI_time][channel].append(max_peak_value)

    Total_N = {}
    Total_N['Control'] = {}
    Total_N['Gabazine'] = {}
    #create a dictionary to hold the total number of cells for each ISI time per channel
    for ISI_time in control_data:
        Total_N['Control'][ISI_time] = {}
        for channel in control_data[ISI_time]:
            Total_N['Control'][ISI_time][channel] = len(control_data[ISI_time][channel])
    for ISI_time in gabazine_data:
        Total_N['Gabazine'][ISI_time] = {}
        for channel in gabazine_data[ISI_time]:
            Total_N['Gabazine'][ISI_time][channel] = len(gabazine_data[ISI_time][channel])


    # Function to calculate mean and standard error
    def calculate_mean_std_error(data):
        mean_data = {}
        std_error_data = {}

        for ISI_time, channels in data.items():
            mean_data[ISI_time] = {}
            std_error_data[ISI_time] = {}
            
            for channel, values in channels.items():
                if values:  # Ensure the list is not empty
                    mean_data[ISI_time][channel] = np.mean(values)
                    std_error_data[ISI_time][channel] = np.std(values) / np.sqrt(len(values))
                else:
                    mean_data[ISI_time][channel] = np.nan
                    std_error_data[ISI_time][channel] = np.nan

        return mean_data, std_error_data

    # Calculate mean and standard error for control and gabazine data
    control_mean, control_std_error = calculate_mean_std_error(control_data)
    gabazine_mean, gabazine_std_error = calculate_mean_std_error(gabazine_data)

    return control_mean, control_std_error, gabazine_mean, gabazine_std_error, Total_N

def calculate_expected_EPSP_stats(expected_EPSP_dict, ISI_times, channels):
    """
    Calculate means and standard errors for compound EPSPs.

    Parameters:
    - compound_EPSP_dict: Dictionary containing compound EPSP peak values.
    - ISI_times: List of ISI times.
    - channels: List of channels.

    Returns:
    - compound_EPSP_means: Dictionary with mean values for each channel and ISI time.
    - compound_EPSP_std_errors: Dictionary with standard errors for each channel and ISI time.
    """
    # Initialize dictionaries to hold the mean and standard error for each ISI and channel for compound EPSPs
    expected_EPSP_means = {channel: np.full(len(ISI_times), np.nan) for channel in channels}
    expected_EPSP_std_errors = {channel: np.full(len(ISI_times), np.nan) for channel in channels}
    expected_EPSP_data = {channel: {ISI_time: [] for ISI_time in ISI_times} for channel in channels}

    # Collect compound EPSP data
    for cell in expected_EPSP_dict:
        for i, channel in enumerate(channels):
            for j, ISI_time in enumerate(ISI_times):
                if channel in expected_EPSP_dict[cell]:
                    if ISI_time in expected_EPSP_dict[cell][channel]:
                        if 'max_peak_value' in expected_EPSP_dict[cell][channel][ISI_time]:
                            current_compound_data = expected_EPSP_dict[cell][channel][ISI_time]['max_peak_value']
                            expected_EPSP_data[channel][ISI_time].append(current_compound_data)

    # Calculate means and standard errors for compound EPSPs
    for channel in channels:
        for j, ISI_time in enumerate(ISI_times):
            if len(expected_EPSP_data[channel][ISI_time]) > 0:
                expected_EPSP_means[channel][j] = np.mean(expected_EPSP_data[channel][ISI_time])
                expected_EPSP_std_errors[channel][j] = np.std(expected_EPSP_data[channel][ISI_time]) / np.sqrt(len(expected_EPSP_data[channel][ISI_time]))

    return expected_EPSP_means,expected_EPSP_std_errors

def plot_EPSP_data(combined_WT_E_I_data, combined_GNB1_E_I_data, ISI_times, channels, channel_names, conditions, colors, fig, ax):
    # Process WT data
    control_mean, control_std_error, gabazine_mean, gabazine_std_error, WT_total_N_dict = process_EPSP_data(combined_WT_E_I_data)
    expected_EPSP_means, expected_EPSP_std_errors = calculate_expected_EPSP_stats(combined_WT_E_I_data['expected_EPSP_amplitudes'], ISI_times, channels)

    #print the N values per ISI time and channel
    for condition in WT_total_N_dict:
        for ISI_time in WT_total_N_dict[condition]:
            for channel in WT_total_N_dict[condition][ISI_time]:
                print('WT')
                print(f'{channel}: {WT_total_N_dict[condition][ISI_time][channel]}')

    # Process GNB1 data
    control_mean_GNB1, control_std_error_GNB1, gabazine_mean_GNB1, gabazine_std_error_GNB1, GNB1_total_N_dict= process_EPSP_data(combined_GNB1_E_I_data)
    expected_EPSP_means_GNB1, expected_EPSP_std_errors_GNB1 = calculate_expected_EPSP_stats(combined_GNB1_E_I_data['expected_EPSP_amplitudes'], ISI_times, channels)

    #print the N values per ISI time and channel
    for condition in GNB1_total_N_dict:
        for ISI_time in GNB1_total_N_dict[condition]:
            for channel in GNB1_total_N_dict[condition][ISI_time]:
                print('GNB1')
                print(f'{channel}: {GNB1_total_N_dict[condition][ISI_time][channel]}')

    x_values = range(len(ISI_times))  # Evenly spaced x-values corresponding to the ISI times

    # Plot WT data
    for i, channel in enumerate(channels):
        # Gather means and errors for Control and Gabazine conditions
        control_means = [control_mean[ISI_time][channel] for ISI_time in ISI_times]
        control_errors = [control_std_error[ISI_time][channel] for ISI_time in ISI_times]
        gabazine_means = [gabazine_mean[ISI_time][channel] for ISI_time in ISI_times]
        gabazine_errors = [gabazine_std_error[ISI_time][channel] for ISI_time in ISI_times]

        # Plotting WT data
        ax[0, i].errorbar(x_values, control_means, yerr=control_errors, fmt='o', color='black', capsize=5)
        ax[0, i].errorbar(x_values, gabazine_means, yerr=gabazine_errors, fmt='o', color='red', capsize=5)
        ax[0, i].plot(x_values, control_means, color='black', linestyle='-', label='Measured - With Inhibition')
        ax[0, i].plot(x_values, gabazine_means, color='red', linestyle='-', label='Measured - No Inhibition')
        ax[0, i].errorbar(x_values, expected_EPSP_means[channel], yerr=expected_EPSP_std_errors[channel], fmt='o', color='blue', capsize=5)
        ax[0, i].plot(x_values, expected_EPSP_means[channel], color='blue', linestyle='-', label='Expected - Linear Summation')
        
        ax[0, i].set_title(f'{channel_names[i]} (WT)')
        ax[0, i].set_xlabel('ISI Time (ms)')
        ax[0, i].set_ylabel('EPSP Amplitude (mV)')
        ax[0, i].set_xticks(x_values)  # Set x-ticks to match the number of ISI times
        ax[0, i].set_xticklabels(ISI_times)  # Label x-ticks with actual ISI times
        ax[0, i].legend()

    # Plot GNB1 data
    for i, channel in enumerate(channels):
        # Gather means and errors for Control and Gabazine conditions
        control_means = [control_mean_GNB1[ISI_time][channel] for ISI_time in ISI_times]
        control_errors = [control_std_error_GNB1[ISI_time][channel] for ISI_time in ISI_times]
        gabazine_means = [gabazine_mean_GNB1[ISI_time][channel] for ISI_time in ISI_times]
        gabazine_errors = [gabazine_std_error_GNB1[ISI_time][channel] for ISI_time in ISI_times]

        # Plotting GNB1 data
        ax[1, i].errorbar(x_values, control_means, yerr=control_errors, fmt='o', color='black', capsize=5)
        ax[1, i].errorbar(x_values, gabazine_means, yerr=gabazine_errors, fmt='o', color='red', capsize=5)
        ax[1, i].plot(x_values, control_means, color='black', linestyle='--', label='Measured - With Inhibition')
        ax[1, i].plot(x_values, gabazine_means, color='red', linestyle='--', label='Measured - No Inhibition')
        ax[1, i].errorbar(x_values, expected_EPSP_means_GNB1[channel], yerr=expected_EPSP_std_errors_GNB1[channel], fmt='o', color='blue', capsize=5)
        ax[1, i].plot(x_values, expected_EPSP_means_GNB1[channel], color='blue', linestyle='--', label='Expected - Linear Summation')
        
        ax[1, i].set_title(f'{channel_names[i]} (GNB1)')
        ax[1, i].set_xlabel('ISI Time (ms)')
        ax[1, i].set_ylabel('EPSP Amplitude (mV)')
        ax[1, i].set_xticks(x_values)  # Set x-ticks to match the number of ISI times
        ax[1, i].set_xticklabels(ISI_times)  # Label x-ticks with actual ISI times
        ax[1, i].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the suptitle
    plt.show()


def get_p_values(E_I_imbalance_data_WT, E_I_imbalance_data_GNB1, ISI_times, channels):
    """
    Calculate p-values for E/I imbalance data between WT and GNB1.

    Parameters:
    - E_I_imbalance_data_WT: Dictionary containing E/I imbalance data for WT.
    - E_I_imbalance_data_GNB1: Dictionary containing E/I imbalance data for GNB1.
    - ISI_times: List of ISI times.
    - channels: List of channels.

    Returns:
    - p_values: Dictionary with p-values for each channel and ISI time.
    """
    p_values = {channel: np.full(len(ISI_times), np.nan) for channel in channels}

    WT_data = {}
    GNB1_data = {}
    for cell in E_I_imbalance_data_WT:
        for ISI_time in E_I_imbalance_data_WT[cell]:
            for channel in E_I_imbalance_data_WT[cell][ISI_time]:
                WT_data.setdefault(ISI_time, {}).setdefault(channel, [])
                WT_data[ISI_time][channel].append(E_I_imbalance_data_WT[cell][ISI_time][channel])
    
    for cell in E_I_imbalance_data_GNB1:
        for ISI_time in E_I_imbalance_data_GNB1[cell]:
            for channel in E_I_imbalance_data_GNB1[cell][ISI_time]:
                GNB1_data.setdefault(ISI_time, {}).setdefault(channel, [])
                GNB1_data[ISI_time][channel].append(E_I_imbalance_data_GNB1[cell][ISI_time][channel])

    for i, channel in enumerate(channels):
        for j, ISI_time in enumerate(ISI_times):
            WT_values = WT_data[ISI_time][channel]
            GNB1_values = GNB1_data[ISI_time][channel]

            #flatten the list of lists
            WT_values = [item for sublist in WT_values for item in sublist]
            GNB1_values = [item for sublist in GNB1_values for item in sublist]

            #exclude values that are empty and greater than 1
            WT_values = [value for value in WT_values if value and value < 1]
            GNB1_values = [value for value in GNB1_values if value and value < 1]

            #calculate p-value
            p_values[channel][j] = stats.ttest_ind(WT_values, GNB1_values).pvalue


            #make correction for bonferroni with alpha and multiple comparisons
            alpha = 0.05
            num_comparisons = 5
            alpha_corrected = alpha / num_comparisons

            #corrected p-values
            corrected_p_values = multipletests(p_values[channel][j], alpha=alpha_corrected, method='bonferroni')[1]
            
    return p_values


import numpy as np

def filter_and_match_data(holding_potentials, estimated_inhibition):
    """
    Filters and matches holding potentials and estimated inhibition data based on common cells.
    Averages holding potentials, handles empty values by filling with NaN, and ensures data consistency.

    Parameters:
    holding_potentials (dict): The dictionary containing holding potentials data.
    estimated_inhibition (dict): The dictionary containing estimated inhibition data.

    Returns:
    tuple: Filtered holding potentials and estimated inhibition as dictionaries.
    """
    filtered_holding_potentials = {}
    filtered_estimated_inhibition = {}

    # Filter holding potentials
    for cell in holding_potentials:
        if cell in estimated_inhibition:
            for ISI_time, channels in holding_potentials[cell].items():
                if cell not in filtered_holding_potentials:
                    filtered_holding_potentials[cell] = {}
                if ISI_time not in filtered_holding_potentials[cell]:
                    filtered_holding_potentials[cell][ISI_time] = {}

                for channel, conditions in channels.items():
                    if channel not in filtered_holding_potentials[cell][ISI_time]:
                        filtered_holding_potentials[cell][ISI_time][channel] = {}

                    for condition, potentials in conditions.items():
                        if condition not in filtered_holding_potentials[cell][ISI_time][channel]:
                            filtered_holding_potentials[cell][ISI_time][channel][condition] = {
                                'unitary_holding_potentials': [],
                                'non_unitary_holding_potentials': []
                            }

                        unitary_potentials = potentials.get('unitary_holding_potentials', [])
                        non_unitary_potentials = potentials.get('non_unitary_holding_potentials', [])

                        # Average and store holding potentials
                        filtered_holding_potentials[cell][ISI_time][channel][condition]['unitary_holding_potentials'].append(
                            np.mean(unitary_potentials) if unitary_potentials else np.nan
                        )
                        filtered_holding_potentials[cell][ISI_time][channel][condition]['non_unitary_holding_potentials'].append(
                            np.mean(non_unitary_potentials) if non_unitary_potentials else np.nan
                        )

    # Filter estimated inhibition
    for cell, isi_data in estimated_inhibition.items():
        if cell in holding_potentials:
            for ISI_time, channels in isi_data.items():
                if ISI_time not in filtered_estimated_inhibition:
                    filtered_estimated_inhibition[ISI_time] = {}

                for channel, inhibition_data in channels.items():
                    if channel not in filtered_estimated_inhibition[ISI_time]:
                        filtered_estimated_inhibition[ISI_time][channel] = []

                    filtered_estimated_inhibition[ISI_time][channel].append(inhibition_data if inhibition_data else np.nan)

    return filtered_holding_potentials, filtered_estimated_inhibition

def plot_inhibition_vs_holding_potential(estimated_inhibition, holding_potentials, label, channel_to_plot, condition_to_plot, color, fig, axs):
    # y values are the estimated inhibitions
    y_values = {}
    # Extract values for each ISI_time and channel
    for ISI_time in estimated_inhibition:
        for channel in estimated_inhibition[ISI_time]:
            if channel == channel_to_plot:
                y_values[ISI_time] = estimated_inhibition[ISI_time][channel]

    # Flatten the list while keeping NaN values
    for ISI_time in y_values:
        flattened_values = []
        for values in y_values[ISI_time]:
            if isinstance(values, list):
                # Flatten the list and keep NaN values
                flattened_values.extend(values)
            else:
                flattened_values.append(values)
        y_values[ISI_time] = flattened_values

    # x values are the holding potentials, remember to average them and that unitary and non-unitary are separate
    x_values = {}
    # Extract values for each ISI_time and channel and condition
    for cell in holding_potentials:
        for ISI_time in holding_potentials[cell]:
            if ISI_time == 300:
                for channel in holding_potentials[cell][ISI_time]:
                    if channel == channel_to_plot:
                        for condition in holding_potentials[cell][ISI_time][channel]:
                            if condition == condition_to_plot:
                                if ISI_time not in x_values:
                                    x_values[ISI_time] = []
                                unitary_values = holding_potentials[cell][ISI_time][channel][condition]['unitary_holding_potentials']
                                x_values[ISI_time].extend(unitary_values)
            else: 
                for channel in holding_potentials[cell][ISI_time]:
                    if channel == channel_to_plot:
                        for condition in holding_potentials[cell][ISI_time][channel]:
                            if condition == condition_to_plot:
                                if ISI_time not in x_values:
                                    x_values[ISI_time] = []
                                non_unitary_values = holding_potentials[cell][ISI_time][channel][condition]['non_unitary_holding_potentials']
                                x_values[ISI_time].extend(non_unitary_values)

    # Plot the data in a scatter plot for each ISI_time as a separate subplot
    for i, ISI_time in enumerate(y_values):
        axs[i].scatter(x_values[ISI_time], y_values[ISI_time], color = color, label = label) 
        axs[i].set_title(f'ISI Time: {ISI_time} ms')
        axs[i].set_ylabel('Estimated Inhibition (%)')
        axs[i].legend()

    axs[-1].set_xlabel('Holding Potential (mV)')
    plt.tight_layout()
    plt.show()


'''Plateau analysis functions'''

# Load the data for plateaus
import os
import pandas as pd


def get_plateau_traces(data_dir):
    data_files = [file for file in os.listdir(data_dir) if file.endswith('.pkl')]
    plateau_traces = {}

    try:
        for data_file in data_files:
            data_file_name = data_file.split('.')[0]
            current_data_df = pd.read_pickle(os.path.join(data_dir, data_file))

            for i in range(len(current_data_df)):
                # Initialize plateau_traces for this file
                if data_file_name not in plateau_traces:
                    plateau_traces[data_file_name] = {}

                entry = current_data_df.iloc[i]

                if 'Theta Stim' in entry['experiment_description']:
                    experiment_desc = entry['experiment_description']

                    # Ensure the experiment description exists as a dictionary
                    if experiment_desc not in plateau_traces[data_file_name]:
                        plateau_traces[data_file_name][experiment_desc] = {}

                    # Assign an index as a key for each trace
                    index_key = f"sweep_{i}"

                    if 'offset_trace' in entry['intermediate_traces']:
                        plateau_traces[data_file_name][experiment_desc][index_key] = entry['intermediate_traces'][
                            'offset_trace']

    except Exception as e:
        print(f"Error processing file {data_file}: {e}")

    return plateau_traces

def get_num_files_with_traces(plateau_traces):
    return len([file for file in plateau_traces if plateau_traces[file]])

def plot_plateau_traces(plateau_traces, title, fig, axes):
    # Filter out empty dictionaries
    non_empty_files = {file: traces for file, traces in plateau_traces.items() if traces}

    print('Files with traces:', non_empty_files.keys())

    fig.suptitle(title, fontsize=16)

    for row, file in enumerate(non_empty_files):
        for experiment in non_empty_files[file]:
            data = non_empty_files[file][experiment]

            # Check if data is a dictionary with multiple sweeps
            if isinstance(data, dict):
                for key in data:
                    data_to_plot = data[key]  # Numeric data for the current sweep
                    time = np.arange(0, len(data_to_plot) * 0.05, 0.05)
                    if 'Perforant' in experiment:
                        axes[row][0].plot(time, data_to_plot, label=f"{key}")
                        axes[row][0].set_title(f'Perforant - {file}')
                        axes[row][0].legend(loc='best')
                    elif 'Schaffer' in experiment:
                        axes[row][1].plot(time, data_to_plot, label=f"{key}")
                        axes[row][1].set_title(f'Schaffer - {file}')
                        axes[row][1].legend(loc='best')
                    elif 'Both Pathway' in experiment:
                        axes[row][2].plot(time, data_to_plot, label=f"{key}")
                        axes[row][2].set_title(f'Both Pathways - {file}')
                        axes[row][2].legend(loc='best')
            else:
                time = np.arange(0, len(data) * 0.05, 0.05)
                # If data is already numeric (array or list), plot directly
                if 'Perforant' in experiment:
                    axes[row][0].plot(time, data, label=file)
                    axes[row][0].set_title(f'Perforant - {file}')
                    axes[row][0].legend(loc='best')
                elif 'Schaffer' in experiment:
                    axes[row][1].plot(time, data, label=file)
                    axes[row][1].set_title(f'Schaffer - {file}')
                    axes[row][1].legend(loc='best')
                elif 'Both Pathway' in experiment and not 'Intracellular' in experiment:
                    axes[row][2].plot(time, data, label=file)
                    axes[row][2].set_title(f'Both Pathways - {file}')
                    axes[row][2].legend(loc='best')

            # Set the same labels for all axes in the row
            for i in range(3):
                axes[row][i].set_xlabel('Time (ms)')
                axes[row][i].set_ylabel('Voltage (mV)')
                axes[row][i].set_xlim(0, 2500)
                axes[row][i].set_ylim(-25, 80)

    # Adjust layout to avoid overlapping of titles and labels
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def calculate_plateau_area(plateau_traces, selected_sweeps):
    stim_start_index = int(500 * 20000 / 1000)
    stim_end_index = int(2200 * 20000 / 1000)

    plateau_areas = {}

    for file, sweeps in selected_sweeps.items():
        if file not in plateau_areas:
            plateau_areas[file] = {}

        for experiment, indices in sweeps.items():
            data = plateau_traces[file][experiment]

            # Check if data is a dictionary (multiple sweeps)
            if isinstance(data, dict):
                for index in indices:
                    sweep_data = data[index][stim_start_index:stim_end_index]

                    # Zero out negative parts of the data
                    sweep_data[sweep_data < 0] = 0

                    # Calculate area under the curve in mV*ms
                    area = np.trapz(sweep_data, dx=0.05)
                    plateau_areas[file][f"{experiment}_sweep_{index}"] = area

            else:
                # Single sweep case
                data = np.array(data)
                data = data[stim_start_index:stim_end_index]
                data[data < 0] = 0

                # Calculate area under the curve in mV*ms
                area = np.trapz(data, dx=0.05)
                plateau_areas[file][f"{experiment}_sweep_0"] = area

    return plateau_areas

# def get_plateau_area_and_plot(plateau_traces):
#     non_empty_files = {file: traces for file, traces in plateau_traces.items() if traces}
#     plateau_areas = {}
#     num_files = len(non_empty_files)
#
#     stim_start_index = int(500 * 20000 / 1000)
#     stim_end_index = int(2200 * 20000 / 1000)
#
#     # Create a figure with subplots, setting the size to 6x4 per subplot
#     fig, axes = plt.subplots(num_files, 1, figsize=(6, 4 * num_files))
#
#     # Handle the case where there's only one file (axes will not be a list in this case)
#     if num_files == 1:
#         axes = [axes]
#
#     for row, file in enumerate(non_empty_files):
#         for experiment in non_empty_files[file]:
#             data = non_empty_files[file][experiment]
#
#             if file not in plateau_areas:
#                 plateau_areas[file] = {}
#
#             # If data is a dictionary (multiple sweeps)
#             if isinstance(data, dict):
#                 # Collect all traces under "Both Pathway"
#                 both_traces = []
#                 if 'Both Pathway' in experiment and not 'Intracellular' in experiment:
#                     for key in data:
#                         current_data = data[key][stim_start_index:stim_end_index]
#                         both_traces.append(current_data)
#
#                     if both_traces:
#                         # Compute the average of the traces
#                         avg_data = np.mean(both_traces, axis=0)
#
#                         # Zero out negative parts of the data
#                         avg_data[avg_data < 0] = 0
#
#                         # Generate the time axis assuming 0.05 ms per sample
#                         time = np.arange(0, len(avg_data) * 0.05, 0.05)
#
#                         # Calculate area under the curve in mV*ms
#                         area = np.trapz(avg_data, dx=1/20000)
#                         plateau_areas[file][experiment] = area
#
#                         # Plot the averaged data with filled area under the curve
#                         axes[row].fill_between(time, avg_data, alpha=0.5, label=f"{file} - Averaged")
#
#             else:
#                 # If data is already numeric (array or list), plot directly
#                 if 'Both Pathway' in experiment and not 'Intracellular' in experiment:
#                     if experiment not in plateau_areas[file]:
#                         plateau_areas[file][experiment] = {}
#
#                     # Zero out negative parts of the data
#                     data = np.array(data)
#                     data[data < 0] = 0
#
#                     # Index the data to the stim start and end
#                     data = data[stim_start_index:stim_end_index]
#
#                     # Generate the time axis assuming 0.05 ms per sample
#                     time = np.arange(0, len(data) * 0.05, 0.05)
#
#                     # Calculate area under the curve in mV*ms
#                     area = np.trapz(data, dx=1/20000)
#                     plateau_areas[file][experiment] = area
#
#                     # Plot the data with filled area under the curve
#                     axes[row].fill_between(time, data, alpha=0.5, label=file)
#
#         # Add legend to each subplot
#         axes[row].legend(loc='upper right')
#         axes[row].set_title(f'File: {file}')
#         axes[row].set_xlabel('Time (ms)')
#         axes[row].set_ylabel('Voltage (mV)')
#
#     plt.tight_layout()
#     plt.show()
#
#     return plateau_areas

def get_plateau_area_and_plot_select_sweeps(plateau_traces, selected_dict):
    non_empty_files = {file: traces for file, traces in plateau_traces.items() if traces}
    plateau_areas = {}
    plateau_traces = [] # List to store the traces for each file
    num_files = len(non_empty_files)

    # Create a figure with subplots, setting the size to 6x4 per subplot
    fig, axes = plt.subplots(num_files, 1, figsize=(6, 4 * num_files))

    # Handle the case where there's only one file (axes will not be a list in this case)
    if num_files == 1:
        axes = [axes]

    for row, file in enumerate(non_empty_files):
        if file in selected_dict:
            experiment = None
            for experiment in non_empty_files[file]:
                if 'Both Pathway' in experiment and not 'Intracellular' in experiment:
                    break
            if experiment is None:
                raise Exception(f"No suitable experiment found for file {file}")
            if file not in plateau_areas:
                plateau_areas[file] = {}
            if experiment not in plateau_areas[file]:
                plateau_areas[file][experiment] = {}
            data = non_empty_files[file][experiment]
            sweep = selected_dict[file]
            #start index is -100 ms from the first value of the sweep
            start_index = int(500 * 20000 / 1000)
            stim_start_index = start_index - int(100 * 20000 / 1000)
            stim_end_index = start_index + int(2200 * 20000 / 1000)
            current_data = data[sweep][stim_start_index:stim_end_index]
            current_data[current_data < 0] = 0
            # Append the traces to the list
            plateau_traces.append(current_data)

            # Generate the time axis assuming 0.05 ms per sample
            time = np.arange(-100, len(current_data) * 0.05 - 100, 0.05)

            # Calculate area under the curve in mV*ms
            area = np.trapz(current_data, dx=1/20000)
            plateau_areas[file][experiment][sweep] = area

            # Plot the data with filled area under curve using the time axis
            axes[row].fill_between(time, current_data, alpha=0.5, label=f"{file} - {sweep}")

        # Add legend to each subplot
        axes[row].legend(loc='upper right')
        axes[row].set_title(f'File: {file}')
        axes[row].set_xlabel('Time (ms)')
        axes[row].set_ylabel('Voltage (mV)')

    # Find the minimum length of all traces
    min_length = min([len(trace) for trace in plateau_traces])

    # Truncate all traces to the same length
    truncated_traces = [trace[:min_length] for trace in plateau_traces]

    # Calculate the average of the truncated traces
    avg_trace = np.mean(truncated_traces, axis=0)
    std_error_trace = np.std(truncated_traces, axis=0) / np.sqrt(len(truncated_traces))

    plt.tight_layout()
    plt.show()

    return plateau_areas, avg_trace, std_error_trace



# def get_plateau_area_and_plot_select_sweeps(plateau_traces, selected_sweeps):
#     non_empty_files = {file: traces for file, traces in plateau_traces.items() if traces}
#     plateau_areas = {}
#     num_files = len(non_empty_files)
#
#     stim_start_index = int(500 * 20000 / 1000)
#     stim_end_index = int(2200 * 20000 / 1000)
#
#     # Create a figure with subplots, setting the size to 6x4 per subplot
#     fig, axes = plt.subplots(num_files, 1, figsize=(6, 4 * num_files))
#
#     # Handle the case where there's only one file (axes will not be a list in this case)
#     if num_files == 1:
#         axes = [axes]
#
#     for row, file in enumerate(non_empty_files):
#         for experiment in non_empty_files[file]:
#             data = non_empty_files[file][experiment]
#
#             if file not in plateau_areas:
#                 plateau_areas[file] = {}
#
#             # Check if data is a dictionary
#             if isinstance(data, dict):
#                 for key in data:
#                     if key in selected_sweeps:
#                         current_data = data[key][stim_start_index:stim_end_index]
#
#                         if 'Both Pathway' in experiment and not 'Intracellular' in experiment:
#                             if experiment not in plateau_areas[file]:
#                                 plateau_areas[file][experiment] = {}
#
#                             # Zero out negative parts of the data
#                             current_data[current_data < 0] = 0
#
#                             # Generate the time axis assuming 0.05 ms per sample
#                             time = np.arange(0, len(current_data) * 0.05, 0.05)
#
#                             # Calculate area under the curve in mV*ms
#                             area = np.trapz(current_data, dx=1/20000)
#                             plateau_areas[file][experiment][key] = area
#
#                             # Plot the data with filled area under curve using the time axis
#                             axes[row].fill_between(time, current_data, alpha=0.5, label=f"{file} - {key}")
#
#             else:
#                 # If data is already numeric (array or list), plot directly
#                 if 'Both Pathway' in experiment and not 'Intracellular' in experiment:
#                     if experiment not in plateau_areas[file]:
#                         plateau_areas[file][experiment] = {}
#
#                     # Zero out negative parts of the data
#                     data = np.array(data)
#                     data[data < 0] = 0
#
#                     # Index the data to the stim start and end
#                     data = data[stim_start_index:stim_end_index]
#
#                     # Generate the time axis assuming 0.05 ms per sample
#                     time = np.arange(0, len(data) * 0.05, 0.05)
#
#                     # Calculate area under the curve in mV*ms
#                     area = np.trapz(data, dx=1/20000)
#                     plateau_areas[file][experiment] = area
#
#                     # Plot the data with filled area under curve using the time axis
#                     axes[row].fill_between(time, data, alpha=0.5, label=file)
#
#         # Add legend to each subplot
#         axes[row].legend(loc='upper right')
#         axes[row].set_title(f'File: {file}')
#         axes[row].set_xlabel('Time (ms)')
#         axes[row].set_ylabel('Voltage (mV)')
#
#     plt.tight_layout()
#     plt.show()
#
#     return plateau_areas

'''Plasiticity Analysis Functions'''

def get_plasticity_traces(dir_path, baseline_starts_list, baseline_ends_list, LTP_induction_starts_list, LTP_induction_ends_list, after_LTP_starts_list):
    data_files = os.listdir(dir_path)
    data_files = [file for file in data_files if file.endswith('.pkl')]

    plasticity_traces = {}
    sorted_data_files = []

    # Sort files by cell date and name
    for data_file in data_files:
        # Read in the data file
        data_df = pd.read_pickle(os.path.join(dir_path, data_file))
        # Get the name of the cell - everything in the file name except '_processed_new.pkl'
        cell_date = data_file.split('_')[0]
        cell_name = data_file.split('_')[1]
        data_name = f"{cell_name}_{cell_date}"
        # Find the date pattern of the data_name and sort the data by date
        date = cell_date
        sorted_data_files.append((date, data_name, data_df))
        
    # Sort data files by date
    sorted_data_files.sort()

    # Iterate through the lists and files
    for (start_time, end_time, after_LTP_start_time, LTP_induction_start, LTP_induction_end), (date, data_file_name, current_data_df) in zip(
        zip(baseline_starts_list, baseline_ends_list, after_LTP_starts_list, LTP_induction_starts_list, LTP_induction_ends_list),
        sorted_data_files):

        if data_file_name not in plasticity_traces:
            plasticity_traces[data_file_name] = {}

        # Create the time array for the entire file
        time_points = []
        first_time = current_data_df['global_wall_clock'].iloc[0]
        for i in range(len(current_data_df)):
            time_point = (current_data_df['global_wall_clock'].iloc[i] - first_time) / 60  # Convert to minutes
            time_points.append(time_point)

        # Extract baseline traces based on provided indices
        baseline_df = current_data_df.iloc[start_time:end_time]
        baseline_traces = {}
        baseline_time_points = []
        for idx, entry in baseline_df.iterrows():
            if 'intermediate_traces' in entry and 'offset_trace' in entry['intermediate_traces']:
                baseline_time_points.append(time_points[idx])
                for channel, trace in entry['intermediate_traces']['offset_trace'].items():
                    if channel not in baseline_traces:
                        baseline_traces[channel] = []
                    baseline_traces[channel].append(trace)  # Store the full trace

        # Extract after LTP traces based on provided indices
        after_LTP_df = current_data_df.iloc[after_LTP_start_time:]
        after_LTP_traces = {}
        after_LTP_time_points = []
        for idx, entry in after_LTP_df.iterrows():
            if 'intermediate_traces' in entry and 'offset_trace' in entry['intermediate_traces']:
                after_LTP_time_points.append(time_points[idx])
                for channel, trace in entry['intermediate_traces']['offset_trace'].items():
                    if channel not in after_LTP_traces:
                        after_LTP_traces[channel] = []
                    after_LTP_traces[channel].append(trace)  # Store the full trace
        
        # Extract LTP induction traces - traces that are between the baseline and after LTP traces
        LTP_induction_df = current_data_df.iloc[LTP_induction_start:LTP_induction_end]
        LTP_induction_traces = []
        for idx, entry in LTP_induction_df.iterrows():
            if 'intermediate_traces' in entry and 'stim_removed_trace' in entry['intermediate_traces']:
                current_trace = entry['intermediate_traces']['stim_removed_trace']
                LTP_induction_traces.append(current_trace)  # Store the full trace

        # Finding max peaks for the first 50 ms per trace and organizing them by channel
        max_peaks_baseline = {
            channel: [np.max(single_trace[:50 * 20000 // 1000]) for single_trace in traces]
            for channel, traces in baseline_traces.items()
        }
        max_peaks_after_LTP = {
            channel: [np.max(single_trace[:50 * 20000 // 1000]) for single_trace in traces]
            for channel, traces in after_LTP_traces.items()
        }

        plasticity_traces[data_file_name] = {
            'baseline_traces': baseline_traces,
            'after_LTP_traces': after_LTP_traces,
            'LTP_traces': LTP_induction_traces,  # Store LTP induction traces as a single list
            'max_peaks_baseline': max_peaks_baseline,
            'max_peaks_after_LTP': max_peaks_after_LTP,
            'baseline_time_points': baseline_time_points,
            'after_LTP_time_points': after_LTP_time_points
        }

    return plasticity_traces

def save_and_plot_traces(traces, save_dir, title):
    num_cells = len(traces)
    fig, ax = plt.subplots(num_cells, 2, figsize=(10, 5*num_cells), sharex=True, sharey=True)

    channel_labels = ['Test Pathway', 'Control Pathway']

    baseline_traces = {}
    for i, cell in enumerate(traces):
        if cell not in baseline_traces:
            baseline_traces[cell] = {}
        if 'baseline_traces' in traces[cell]:
            for j, channel in enumerate(traces[cell]['baseline_traces']):
                label = channel_labels[j] if j < len(channel_labels) else f'Channel {j+1}'  # Ensure labeling according to channel

                for trace in traces[cell]['baseline_traces'][channel]:
                    ax[i, j].set_title(f'{cell} - {label}')
                    ax[i, j].set_ylabel('Voltage (mV)')
                    ax[i, j].set_xlabel('Time (ms)')

                    if channel not in baseline_traces[cell]:
                        baseline_traces[cell][channel] = []

                    baseline_traces[cell][channel].append(trace)
    
    
    average_baseline_traces = {}
    for cell in baseline_traces:
        if cell not in average_baseline_traces:
            average_baseline_traces[cell] = {}
        for channel in baseline_traces[cell]:
            average_baseline_traces[cell][channel] = np.mean(baseline_traces[cell][channel], axis=0)

    for idx, cell in enumerate(average_baseline_traces):
        for j, channel in enumerate(average_baseline_traces[cell]):
            current_time = np.arange(0, len(average_baseline_traces[cell][channel]) * 0.05, 0.05)
            ax[idx, j].plot(current_time, average_baseline_traces[cell][channel], color='black', linewidth=2)

    After_LTP_traces = {}
    for a, cell in enumerate(traces):
        if cell not in After_LTP_traces:
            After_LTP_traces[cell] = {}
        if 'after_LTP_traces' in traces[cell]:
            # print(sucrose_plasticity_traces_no_delay[cell]['after_LTP_traces'])
            for b, channel in enumerate(traces[cell]['after_LTP_traces']):
                for trace in traces[cell]['after_LTP_traces'][channel]:
                    if channel not in After_LTP_traces[cell]:
                        After_LTP_traces[cell][channel] = []

                    After_LTP_traces[cell][channel].append(trace)

    average_after_LTP_traces = {}
    for cell in After_LTP_traces:
        if cell not in average_after_LTP_traces:
            average_after_LTP_traces[cell] = {}
        for channel in After_LTP_traces[cell]:
            average_after_LTP_traces[cell][channel] = np.mean(After_LTP_traces[cell][channel], axis=0)

    for idx, cell in enumerate(average_after_LTP_traces):
        for j, channel in enumerate(average_after_LTP_traces[cell]):
            current_time = np.arange(0, len(average_after_LTP_traces[cell][channel]) * 0.05, 0.05)
            ax[idx, j].plot(current_time, average_after_LTP_traces[cell][channel], color='red', linewidth=2)


    #save the figure as a .png file and .svg file
    dir_save_path = save_dir
    fig.savefig(f'{dir_save_path}{title}.png', dpi = 300)
    fig.savefig(f'{dir_save_path}{title}.svg', dpi = 300)
#plot the LTP induction traces

def plot_LTP_induction_traces(traces, save_dir, title):
    num_cells = len(traces)

    fig, ax = plt.subplots(num_cells, 1, figsize=(10, 5*num_cells), sharex=True, sharey=True)

    LTP_induction_traces = {}
    if len(traces) > 1:
        for i, cell in enumerate(traces):
            if 'LTP_traces' in traces[cell]:
                for trace in traces[cell]['LTP_traces']:
                    current_time = np.arange(0, len(trace) * 0.05, 0.05)
                    # ax[i].plot(current_time, trace, color='red')
                    ax[i].set_title(f'{cell} - LTP Induction')
                    ax[i].set_ylabel('Voltage (mV)')
                    ax[i].set_xlabel('Time (ms)')
                    if cell not in LTP_induction_traces:
                        LTP_induction_traces[cell] = []

                    LTP_induction_traces[cell].append(trace)
    else:
        cell = list(traces.keys())[0]
        for i, trace in enumerate(traces[cell]['LTP_traces']):
            current_time = np.arange(0, len(trace) * 0.05, 0.05)
            # ax.plot(current_time, trace, color='red')
            ax.set_title(f'{cell} - LTP Induction')
            ax.set_ylabel('Voltage (mV)')
            ax.set_xlabel('Time (ms)')
            if cell not in LTP_induction_traces:
                LTP_induction_traces[cell] = []

            LTP_induction_traces[cell].append(trace)

    average_LTP_induction_traces = {}
    for cell in LTP_induction_traces:
        average_LTP_induction_traces[cell] = np.mean(LTP_induction_traces[cell], axis=0)

    if len(traces) > 1:
        for idx, cell in enumerate(average_LTP_induction_traces):
            current_time = np.arange(0, len(average_LTP_induction_traces[cell]) * 0.05, 0.05)
            ax[idx].plot(current_time, average_LTP_induction_traces[cell], color='black', linewidth=2)
    else:
        current_time = np.arange(0, len(average_LTP_induction_traces[cell]) * 0.05, 0.05)
        ax.plot(current_time, average_LTP_induction_traces[cell], color='black', linewidth=2)

    #save the figure as a .png file and .svg file
    dir_save_path = save_dir

    fig.savefig(f'{dir_save_path}{title}.png', dpi = 300)
    fig.savefig(f'{dir_save_path}{title}.svg', dpi = 300)

    


