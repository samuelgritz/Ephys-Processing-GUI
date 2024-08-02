import numpy as np
import pandas as pd
import matplotlib as mpl
import yaml
import os

import tkinter as tk
from tkinter import ttk 
from tkinter import simpledialog, filedialog, ttk, messagebox

import matplotlib.figure as figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk

# Import custom functions
import scipy.signal as signal
from scipy.signal import find_peaks

#Updates
#TODO: Add all check boxes automatically to the analysis_steps frame and the user can choose these check boxes to run the analysis, the check marks will be populate the analysis_steps_list
#Then the order of the analysis_steps_list will determine the order of the analysis and will be run in that order and be applied to every trace every time they are pressed

#TODO: analyze the peaks of offset data --> if you want to plot the peaks --> it must be from offset data

#TODO: Fix multiple check buttons being selected at the same time

#TODO add the feature to analyze EPSP amplitudes - plot these peaks on the graph
#TODO add feature to fill in the .yaml file metadata values:

#TODO create another dataframe for plotting purposes

#fixes: 
#TODO: Fix the issue with plotting all the offset traces - make sure it offsets for each trace and plots each trace
#TODO: Fix the issue of plotting only the mean, not the offset mean if offset is pressed


# Set matplotlib configurations
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.size'] = 12.
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['text.usetex'] = False

#Add user entry boxes for all values that need to be set or maintained

def read_from_yaml(file_path, Loader=None):
    """
    Import a python dict from .yaml
    :param file_path: str (should end in '.yaml')
    :param Loader: :class:'yaml.Loader'
    :return: dict
    """
    if Loader is None:
        Loader = yaml.FullLoader
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))

#TODO: make this take in AP_peaks that have been already populated in the data_frame, Analyze by sweep and not traces
def analyze_AP_AHP_properties_trace(trace, time, AP_peak_idx, stim_end, stim_length, acquisition_frequency, Vm_rest_start, Vm_rest_end, AP_peak_window_start, AP_peak_window_end):
    #debugging
    import matplotlib.pyplot as plt
    # plt.figure()
    #Make a copy of the current trace to avoid modifying the original trace
    current_trace = trace.copy()
    time = time.copy()
    # plt.plot(time, current_trace, color='black')

    #Resting membrane potential
    Vm_rest = np.mean(current_trace[Vm_rest_start:Vm_rest_end])

    #AP properties
    AP_number = []
    AP_firing_rate = []
    AP_traces = []
    AP_thresholds = []
    AP_threshold_indices = []
    AP_ISI_list = []
    AP_sizes = []
    AP_halfwidth_list = []

    #AHP properties
    AHP_indices = []
    AHP_sizes = []
    AHP_halfwidth_list = []
    
    # print('AP_peak_idx:', AP_peak_idx)

    #if AP_peak_idx is a single value, then convert it to a list
    if type(AP_peak_idx) == int:
        AP_peak_idx = [AP_peak_idx]

    if len(AP_peak_idx) == 0:
        AP_number = len(AP_peak_idx)
        AP_firing_rate = AP_number/(stim_length/1000)

        print('AP_number:', AP_number)
        print('AP_firing_rate:', AP_firing_rate)
    else:
        AP_number = len(AP_peak_idx)
        AP_firing_rate = AP_number/(stim_length/1000)

        print('AP_number:', AP_number)
        print('AP_firing_rate:', AP_firing_rate)
    
    # Define duration of time before and after the AP peak
    AP_peak_window_start = int(AP_peak_window_start * acquisition_frequency/1000) #duration of time before the AP
    AP_peak_window_end = int(AP_peak_window_end * acquisition_frequency/1000) #duration of time after the AP

    # #Calculate the time between the peaks
    spike_times = time[(AP_peak_idx)]

    prev_peak_idx = 0

    for i, peak_idx in enumerate(AP_peak_idx):
        AP_ISI = spike_times[i] - spike_times[i-1] if i > 0 else np.NaN
        if i > 0:
            AP_ISI_list.append(AP_ISI)
        
        AP_peak_voltage = current_trace[peak_idx]

        window_start = peak_idx - AP_peak_window_start # default value of window start

        temporal_peak_offset = int(np.round(0.5*acquisition_frequency/1000)) # pick an index that is 0.5ms after the previous peak_idx
        #pick the larger of the two values between the previous peak + offset or the windoe start
        # Assuming prev_peak_idx and temporal_peak_offset are numpy arrays
        indices = np.where(prev_peak_idx + temporal_peak_offset > window_start)[0]
        if indices.size > 0:
            window_start = np.max(prev_peak_idx[indices] + temporal_peak_offset[indices])
        # window_start = max(prev_peak_idx+temporal_peak_offset, window_start)#avoid measuring the previous AP in the current window if the previous peak is within the window then pick a value that start just after the previous peak
        window_end = peak_idx + AP_peak_window_end

        # Find the AP threshold - the greatest rate of change in the AP trace (second derivative or third derivative)
        AP_trace = current_trace[window_start:window_end]
        AP_traces.append(AP_trace)

        #try the second derivative
        try:
            derivative = np.diff(AP_trace, n=2)
            current_derivative = derivative.copy()
            # filter the derivative to smooth the trace
            derivative_filtered = signal.savgol_filter(current_derivative, 15, 3, mode='mirror')
            try:
                threshold_idx_in_AP_trace = find_peaks(derivative_filtered, height=0.5)[0][0] # Find peaks in the filtered derivative
            except:
                threshold_idx_in_AP_trace = 0
            AP_threshold_idx = window_start + threshold_idx_in_AP_trace # Get the threshold index in the original trace
            AP_threshold_indices.append(AP_threshold_idx)
            AP_threshold = current_trace[AP_threshold_idx] # Get the threshold value from the trace
            if AP_threshold > 0: # Fix the syntax error here
                AP_threshold = np.NaN

            AP_thresholds.append(AP_threshold)

        #if second derivative does not work, try the first derivative
        except:
            derivative = np.diff(AP_trace)
            current_derivative = derivative.copy()
            # filter the derivative to smooth the trace
            derivative_filtered = signal.savgol_filter(current_derivative, 15, 3, mode='mirror')
            try:
                threshold_idx_in_AP_trace = find_peaks(derivative_filtered, height=0.5)[0][0] # Find peaks in the filtered derivative
            except:
                threshold_idx_in_AP_trace = 0
            threshold_idx_in_AP_trace = find_peaks(derivative_filtered, height=0.5)[0][0]
            AP_threshold_idx = window_start + threshold_idx_in_AP_trace
            AP_threshold_indices.append(AP_threshold_idx)
            AP_threshold = current_trace[AP_threshold_idx]
            if AP_threshold > 0: # Fix the syntax error here

                AP_threshold = np.NaN

            AP_thresholds.append(AP_threshold)

        try:
            AP_peak_voltage = current_trace[peak_idx] # peak in mV
            AP_size = AP_peak_voltage - AP_threshold # AP in mV
            AP_sizes.append(AP_size)
        except:
            AP_size = np.NaN
            AP_sizes.append(np.NaN)

        try:
            #Action Potential Halfwidths - halfwidth center point is from the center of the AP to the threshold, width is the distance on both sides of the center point
            AP_halfwidth_voltage = AP_threshold + (AP_size)/2 # halfwidth in mV; AP_peak_voltage - AP_threshold = voltage at the center line, thus add threshold (smaller value) to get the distance from center to the threshold
            AP_halfwidth_start_idx = np.where(AP_trace >= AP_halfwidth_voltage)[0][0]
            AP_halfwidth_points = np.where(AP_trace[AP_halfwidth_start_idx:] <= AP_halfwidth_voltage)[0][0] # starting from the halfwidth start, cut off the beginning artifact of the start of the trace, find the first time the voltage is slightly smaller than haldwidth start, as the value on both sides can not be exactly the same
            AP_halfwidth = AP_halfwidth_points/acquisition_frequency * 1000 # halfwidth in ms
            AP_halfwidth_list.append(AP_halfwidth)
            
            # Get the mirror of the AP trace using (-) sign - for After Hyperpolarization
            #if the index is past time 800ms, then set the window end to the end of the trace
            if window_end > int(stim_end * acquisition_frequency/1000):
                window_end = len(current_trace)
            AHP_trough_idx = np.argmin(current_trace[peak_idx:window_end]) + peak_idx # starting from the peak_idx, find the first trough (negative peak)
            # print('AHP_trough_idx:', AHP_trough_idx)
            AHP_indices.append(AHP_trough_idx)
         
            AHP_voltage = current_trace[AHP_trough_idx] # AHP in mV
            # print('AHP_voltage:', AHP_voltage)
            AHP_size = AP_threshold - AHP_voltage # AHP size in mV 
            AHP_sizes.append(AHP_size)
        except:
            AHP_size = np.NaN
            AHP_sizes.append(np.NaN)
            AHP_indices.append(np.NaN)

        try:
            #After Hyperpolarization Halfwidths
            AHP_halfwidth_voltage = (AP_threshold + AHP_voltage)/2 # AHP halfwidth in mV; AP_threshold - AHP_voltage = the voltage at the center line, thus add the AHP voltage (smaller value) to get the distance from center to the AHP voltage
            #find where values are smaller than the haldwidth voltage and get the index of the first value
            # print(AHP_halfwidth_voltage)
            #find this relative to the beginning of the AHP trace
            AHP_halfwidth_start_idx = peak_idx + np.where(current_trace[peak_idx:window_end] <= AHP_halfwidth_voltage)[0][0] 
            # print('AHP_halfwidth_start_idx:', AHP_halfwidth_start_idx)

            # if len(np.where(current_trace[AHP_halfwidth_start_idx:]>=AHP_halfwidth_voltage)[0]) == 0: #if the AHP halfwidth voltage is not found in the trace, set the halfwidth to NaN
            #     AHP_halfwidth = np.NaN
            # else:
            #where does the trace go above the AHP halfwidth voltage is the end so take that index and add to start index to get the width        
            # print(np.where(current_trace[AHP_halfwidth_start_idx:] >= AHP_halfwidth_voltage)[0])    
            AHP_halfwidth_end_idx = np.where(current_trace[AHP_halfwidth_start_idx:] >= AHP_halfwidth_voltage)[0][0] # starting from the halfwidth start, cut off the beginning artifact of the start of the trace, find the first time the voltage is slightly larger than haldwidth start, as the value on both sides can not be exactly the same
            # print('AHP_halfwidth_end_idx:', AHP_halfwidth_end_idx)
            AHP_halfwidth = (AHP_halfwidth_end_idx - AHP_halfwidth_start_idx)/acquisition_frequency * 1000 # AHP halfwidth in ms
            # print('AHP_halfwidth:', AHP_halfwidth)
            AHP_halfwidth_list.append(AHP_halfwidth)
        except:
            AHP_halfwidth = np.NaN
            AHP_halfwidth_list.append(np.NaN)
        

    AP_properties_spikes_dict = {
        'AP_threshold (mV)': AP_thresholds,
        'AP threshold indices': AP_threshold_indices,
        'Vm_rest (mV)': Vm_rest,
        'AP_firing_rate (Hz)': AP_firing_rate,
        'AP_number': AP_number,
        'AP_ISI_time (ms)': AP_ISI_list,
        'AHP_indices': AHP_indices,
        'AP_halfwidth (ms)': AP_halfwidth_list,
        'AP_size (mV)': AP_sizes,
        'AHP_halfwidth (ms)': AHP_halfwidth_list,
        'AHP_size (mV)': AHP_sizes}
    
    return AP_properties_spikes_dict

def interp_spikes(AP_peak_window_start, AP_peak_window_end, spike_width, acquisition_frequency, voltage_trace, time, height):    
    #want to interp from spike onsets to interp_ends
    spike_onsets = []
    interp_ends = []

    height = height

    voltage_trace = voltage_trace.copy()

    AP_peaks =  find_peaks(voltage_trace, height=height)[0]
    
    prev_peak_idx = 0 
    AP_peaks = AP_peaks.copy()
    try:
        for peak_idx in AP_peaks:
        
            window_start = peak_idx - int(AP_peak_window_start*acquisition_frequency/1000)
            temporal_peak_offset = int(np.round(0.5*acquisition_frequency/1000)) # pick an index that is 0.5ms after the previous peak_idx
            indices = np.where(prev_peak_idx + temporal_peak_offset > window_start)[0]
            if indices.size > 0:
                window_start = np.max(prev_peak_idx[indices] + temporal_peak_offset[indices])
            window_end = peak_idx + int(AP_peak_window_end*acquisition_frequency/1000)
            AP_trace = voltage_trace[window_start:window_end]

            #try the second derivative
            derivative = np.diff(AP_trace, n=2)
            current_derivative = derivative.copy()
            derivative_filtered = signal.savgol_filter(current_derivative, 15, 3, mode='mirror')
            try:
                threshold_idx_in_AP_trace = find_peaks(derivative_filtered, height=0.5)[0][0] # Find peaks in the filtered derivative
                # print('threshold_idx_in_AP_trace:', type(threshold_idx_in_AP_trace))
                AP_threshold_idx = window_start + threshold_idx_in_AP_trace # Get the threshold index in the original trace
                AP_threshold_idx -= int(spike_width*acquisition_frequency/1000)
                # print('peak_idx:', type(peak_idx))
                # print('AP_threshold_idx:', type(AP_threshold_idx))
            except Exception as e:
                # print("Error in interp_spikes:", e)
                # threshold_idx_in_AP_trace
                # AP_threshold_idx = window_start + threshold_idx_in_AP_trace # Get the threshold index in the original trace
                AP_threshold_idx = peak_idx - (int(spike_width*acquisition_frequency/1000) + (int(4 * acquisition_frequency/1000)) )
            
            spike_onsets.append(AP_threshold_idx)
        
        for i, idx in enumerate(spike_onsets):
            start = AP_peaks[i] 
            baseline = np.mean(voltage_trace[(idx - int(0.1 * acquisition_frequency/1000)) : idx]) #baseline is 0.1ms before the spike onset
            max_end = idx + int(AP_peak_window_end*acquisition_frequency/1000)
            min_idx = np.argmin(voltage_trace[start : max_end]) + start

            for end in range(start, max_end):
                if voltage_trace[end] <= baseline:
                    break
                if end == min_idx:
                    break
                try:
                    if end == spike_onsets[i + 1]:
                        break
                except:
                    pass
            interp_ends.append(end)

        for i, idx in enumerate(spike_onsets):
            end_index = interp_ends[i]
            voltage_trace[idx + 1 : end_index] = np.nan

        is_nan_bool_array = np.isnan(voltage_trace)
        voltage_trace[is_nan_bool_array] = np.interp(time[is_nan_bool_array], time[~is_nan_bool_array], voltage_trace[~is_nan_bool_array])
    
        return voltage_trace
    
    except:
        pass

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
            input_resistance = abs(delta_voltage / -delta_current) * 10**-6 #convert to megaohms
            return input_resistance
        except Exception as e:
            print("Error calculating input resistance:", e)
            pass

def calculate_voltage_sag(experiment_type, voltage_trace, acquisition_frequency, start_time, end_time):
    #TODO fix the voltage sag calculation
    try:
        if experiment_type != 'voltage_sag':
            #have a pop up that says the experiment type is not voltage sag
            messagebox.showinfo("Error", "The experiment type is not voltage sag, please select the correct experiment type")
        else:
            #convert the start, end time and acquisition frequency to integers
            start_time = int(start_time * acquisition_frequency/1000)
            end_time = int(end_time * acquisition_frequency/1000) 
            baseline_start = start_time - int(10 * acquisition_frequency/1000)
            #calculate the baseline voltage
            baseline_voltage = np.mean(voltage_trace[baseline_start:start_time])
            # print('baseline_voltage:', baseline_voltage)
            # Find the minimum voltage (sag minimum) during the hyperpolarizing step
            min_voltage_during_sag = min(voltage_trace)
            # print('min_voltage_during_sag:', min_voltage_during_sag)
            # Find the steady-state voltage at the end of the hyperpolarizing step 50ms before the end of the step
            steady_state_voltage = np.mean(voltage_trace[end_time - int(10 * acquisition_frequency/1000):end_time])
            # print('steady_state_voltage:', steady_state_voltage)
            # Calculate the sag ratio
            sag_ratio = (steady_state_voltage - min_voltage_during_sag) / (baseline_voltage - min_voltage_during_sag)
            sag_ratio = abs(sag_ratio) * 100

            return sag_ratio
        
    except Exception as e:
        print("Error calculating voltage sag:", e)
        pass

# Function to remove stimulation artifacts
def remove_artifacts_custom(data, artifact_times, acquisition_frequency, delete_start_stim, delete_end_stim):
    processed_data = np.copy(data)  # Make a copy of the data to avoid modifying the original
    for artifact_time in artifact_times:
        current_stim_index = int(artifact_time * acquisition_frequency / 1000)
        delete_start_index = max(0, current_stim_index - int(delete_start_stim * acquisition_frequency / 1000))
        delete_end_index = min(len(processed_data), current_stim_index + int(delete_end_stim * acquisition_frequency / 1000))
        current_window = np.arange(delete_start_index, delete_end_index)
        processed_data[current_window] = np.nan
        processed_data = np.interp(np.arange(0, len(processed_data)), np.arange(0, len(processed_data))[~np.isnan(processed_data)], processed_data[~np.isnan(processed_data)])

    return processed_data

def remove_artifacts_plateau(data, time, acquisition_frequency, delete_start_stim, delete_end_stim, height):
        '''Remove stimulation artifacts from plateau experiments'''
        try:
            current_trace = np.copy(data)
            #get the forth derivative using np.diff
            forth_derivative = np.diff(current_trace, n=4) 
            #find AP peaks
            AP_peaks = find_peaks(current_trace, height=height)[0]
            #get the time of the action potential peak
            AP_peaks_times = time[AP_peaks]
            #find all peaks in the forth derivative
            all_peaks = find_peaks(forth_derivative, height=1.5)[0]
            #If a peak occurs 1ms from an AP_peak, remove it from all peaks
            for peak in all_peaks:
                if any(abs(peak - AP_peaks) < int(1 * acquisition_frequency/1000) + 1):
                    all_peaks = np.delete(all_peaks, np.where(all_peaks == peak))
            #redefine the stim times
            stim_times = time[all_peaks]
            #create a processed trace
            processed_trace = np.copy(current_trace)
            #remove the stimulation artifacts
            processed_trace = remove_artifacts_custom(processed_trace, stim_times, acquisition_frequency, delete_start_stim, delete_end_stim)
            
            return processed_trace
        except Exception as e:
            print("Error in remove_artifacts_plateau:", e)
            return None
    
def find_peaks_in_window(voltage_trace, delete_start_time, window_size, acquisition_frequency):
    # Find the peaks in a window of a single trace
    delete_start_index = int(delete_start_time * acquisition_frequency / 1000)
    current_trace = voltage_trace

    window_size = 100 
    current_window = current_trace[delete_start_index: int ( delete_start_index + (window_size * acquisition_frequency / 1000) )]

    max_peak_value = []
    max_peak_idx = []
    peaks, _ = find_peaks(current_window, height=1)
    if peaks.size > 0:
        # Get the max peak index
        max_peak = max(current_window[peaks])
        max_peak_index = np.where(current_window == max_peak)[0][0]
        max_peak_value.append(max_peak)

        # Get the index of the max peak
        max_peak_idx.append(max_peak_index + int(delete_start_index * acquisition_frequency / 1000))

    return max_peak_value, max_peak_idx

#TODO create data_class for loading the data and storing the data and manipulating the data can call outside of the GUI, the init takes in file path and the data_dict
# Main class
class GUI(object):

    """Main class for the GUI. Instances of this GUI receive default metadata and contain methods for processing data and plotting"""
    def __init__(self, sweep_properties, cell_meta_data_fields, cell_properties, 
                 analysis_types, defaults):
        
        """Initialize the GUI with the given parameters
        Args:
        params_root: Tkinter root for the processing options and steps
        plot_data_root: Tkinter root for the data plots
        plot_r_inp_root: Tkinter root for the input resistance plots
        sweep_properties: List of sweep properties
        cell_meta_data_fields: List of cell metadata fields
        cell_properties: List of cell properties
        analysis_types: List of analysis types
        defaults: Dictionary of default values for the GUI
        """
        self.sweep_properties = sweep_properties
        self.cell_meta_data_fields = cell_meta_data_fields
        self.cell_properties = cell_properties
        self.analysis_types = analysis_types
        self.defaults = defaults
        self.data_df = self.open_file()
        self.data_df['analysis_dict'] = [{} for _ in range(len(self.data_df))]
        self.data_df['intermediate_traces'] = [{} for _ in range(len(self.data_df))]
        self.data_df['current_trace_key'] = ['raw' for _ in range(len(self.data_df))]
        self.analysis_steps_list = []
        self.processed_data_label = {}
        self.plot_widgets = {}

        #create experiment_type lists for conditionals
        self.EPSP_list = []
        self.plateau_list = []
        #for EPSP_list have it be instances where num_pulses is equal to 2 or 3 but less than 5
        #for plateau_list have it be instances where num_pulses 5 or greater
        #first check if the key in self.defaults['analysis'] has 'channel' in it
        #if it does, then check if the value of 'num_pulses' is 2 or 3
        #if it is, then add the key to EPSP_list
        #if the value of 'num_pulses' is 5 or greater, then add the key to plateau_list
        for key in self.defaults['analysis'].keys():
            #if the string channel is contained in any of the nested keys under key
            if any('channel' in nested_key for nested_key in self.defaults['analysis'][key].keys()):
                for channel in self.defaults['analysis'][key].keys():
                    if self.defaults['analysis'][key][channel]['num_pulses'] == 2 or self.defaults['analysis'][key][channel]['num_pulses'] == 3:
                        self.EPSP_list.append(key)
                    elif self.defaults['analysis'][key][channel]['num_pulses'] >= 5:
                        self.plateau_list.append(key)
        
        #make sure that the EPSP_list and plateau_list are unique
        self.EPSP_list = list(set(self.EPSP_list))
        self.plateau_list = list(set(self.plateau_list))

        #store temporary data values 
        self.ISI_time = []
        self.offset_traces = {}
        self.average_data = {}
        self.average_offset_traces = {}
        self.temp_data = None
        self.current_index = 0

        #Root for dealing with singles traces analysis, plotting, and data manipulation
        self.params_root = tk.Toplevel()
        self.params_root.title("Electrophysiology Visualization Tool Parameters")

        #Root for dealing with multiple traces analysis, plotting, and data manipulation
        # self.multiple_params_root = tk.Toplevel()
        # self.multiple_params_root.title("Multiple Trace Analysis")

        #Plotting roots
        self.plot_data_root = tk.Toplevel()
        self.plot_data_root.title("Plot Data")

        self.plot_metadata_root = tk.Toplevel()
        self.plot_metadata_root.title("Plot Metadata")

        self.plot_r_inp_root = tk.Toplevel()
        self.plot_r_inp_root.title("Plot Input Resistance")

        #set up the structure of the GUI for single trace analysis
        self.load_navigate_frame = ttk.Frame(self.params_root, borderwidth=1, relief=tk.GROOVE)
        self.export_frame = ttk.Frame(self.params_root, borderwidth=0, relief=tk.GROOVE)
        self.analysis_steps_frame = ttk.Frame(self.params_root, borderwidth=1, relief=tk.GROOVE)
        self.experiment_metadata_frame = ttk.Frame(self.params_root, borderwidth=1, relief=tk.GROOVE)

        self.load_navigate_frame.grid(row=0, column=0, sticky='nsew')
        self.export_frame.grid(row=0, column=1, sticky='nsew')
        self.analysis_steps_frame.grid(row=1, column=0, sticky='nsew')
        self.experiment_metadata_frame.grid(row=1, column=1, sticky='nsew')

        #set up the structure of the GUI for multiple trace analysis
        # self.multiple_load_navigate_frame = ttk.Frame(self.multiple_params_root, borderwidth=1, relief=tk.GROOVE)
        # self.multiple_load_navigate_frame.grid(row=0, column=0, sticky='nsew')

        #plotting on seperate frames
        self.plot_data_frame = ttk.Frame(self.plot_data_root, borderwidth=0, relief=tk.GROOVE)
        self.plot_metadata_frame = ttk.Frame(self.plot_metadata_root, borderwidth=0, relief=tk.GROOVE)
        self.plot_r_inp_frame = ttk.Frame(self.plot_r_inp_root, borderwidth=0, relief=tk.GROOVE)

        self.plot_data_frame.grid(row=0, column=0, sticky='nsew')
        self.plot_metadata_frame.grid(row=0, column=0, sticky='nsew')
        self.plot_r_inp_frame.grid(row=0, column=0, sticky='nsew')

        #check box boolean states
        self.clear_plot = tk.BooleanVar(value=True)
        #FI and cell properties
        self.detect_spikes_check = tk.BooleanVar(value=False)
        self.AP_AHP_analysis_check = tk.BooleanVar(value=False)

        #dictionary to store boolean values based on trace key - this will be dynamically appended to 
        self.plotting_checks = {}
        self.label_checks = {}

        #dictionary to store the check buttons for plotting and labeling
        self.plot_check_buttons = {}
        self.label_check_buttons = {}

        self.stim_removal_check = tk.BooleanVar(value=False)
        self.noise_removal_check = tk.BooleanVar(value=False)
        self.interpolated_spikes_check = tk.BooleanVar(value=False)
        self.offset_traces_check = tk.BooleanVar(value=False)
        self.partition_traces_check = tk.BooleanVar(value=False)

        #preprocessing for EPSP and plateau
        self.stim_removal_pressed = False
        self.noise_removal_pressed = False
        self.interpolated_spikes_pressed = False
        #postprocessing for EPSP and plateau
        self.offset_traces_pressed = False
        self.partition_traces_pressed = False

        #check buttons for plotting labels and plots
        self.plot_labels_legend_list = []
        self.raw_label_toggle = tk.BooleanVar(value=False)
        self.processed_label_toggle = tk.BooleanVar(value=False)

        #toggle to plot the raw data
        self.raw_data_plot_toggle = tk.BooleanVar(value=True)
        #check for plotting the average and the offset
        self.average_plot_toggle = tk.BooleanVar(value=False)
        self.offset_plot_toggle = tk.BooleanVar(value=False)
        self.offset_average_plot_toggle = tk.BooleanVar(value=False)

        #toggle for the offset and average traces
        # self.plot_offset_traces_pressed = False
        self.average_traces_pressed = tk.BooleanVar(value=False)

        #load the frames for the GUI for single trace analysis
        self.widgets_load_navigate_frame()
        self.widgets_export_frame()
        self.widgets_analysis_steps_frame()
        self.widgets_experiment_metadata_frame()
        self.widgets_plot_data_frame()
        self.widgets_plot_metadata_frame()
        self.widgets_plot_r_inp_frame()

        # #load the frames for the GUI for multiple trace analysis
        # self.widgets_multiple_load_navigate_frame()

        #run the mainloop to keep the GUI running
        self.params_root.mainloop()
        # self.multiple_params_root.mainloop()
        self.plot_data_root.mainloop()
        self.plot_metadata_root.mainloop()
        self.plot_r_inp_root.mainloop()
    
    def widgets_load_navigate_frame(self):
        #label for load and navigate frame
        self.load_navigate_label = ttk.Label(self.load_navigate_frame, text="Load and Navigate Data", font=('Arial', 16, 'bold'))
        self.load_navigate_label.grid(row=0, column=0, sticky='nsew')
        #label for select index
        self.select_index_label = ttk.Label(self.load_navigate_frame, text="Select Index")
        self.select_index_label.grid(row=1, column=0, sticky='nsew')

        #add user input box for index 
        self.index_entry = ttk.Entry(self.load_navigate_frame, width=10)
        #set the default value to 0 if zero exists otherwise set to 1
        self.index_entry.insert(0, '0') 
        self.index_entry.grid(row=2, column=0, sticky='nsew')

        # #entry to select a range of indices
        # #label
        # self.select_index_multiple_label = ttk.Label(self.load_navigate_frame, text="Select Multiple Indices")
        # self.select_index_multiple_label.grid(row=1, column=1, sticky='nsew')
        # self.index_entry_multiple = ttk.Entry(self.load_navigate_frame, width=10)
        # self.index_entry_multiple.insert(0, '40,41,42,43,44') 
        # self.index_entry_multiple.grid(row=2, column=1, sticky='nsew')

        #create the first index for plotting
        if self.data_df.experiment_index.iloc[0] == 0:
            self.start_index = 0
        elif self.data_df.experiment_index.iloc[1] == 1:
            self.start_index = 1

        # Dropdown menu to select experiment type
        self.current_experiment_type = self.data_df['stim_type'].iloc[int(self.index_entry.get())]
        self.experiment_list = list(self.defaults['analysis'].keys())
        self.experiment_type_var = tk.StringVar(self.load_navigate_frame)
        self.experiment_type_dropdown = ttk.Combobox(self.load_navigate_frame, textvariable=self.experiment_type_var, values=self.experiment_list)
        #populate the first value of the dropdown menu with the experiment_list[1]
        self.experiment_type_var.set(self.experiment_list[0])
        self.experiment_type_dropdown.grid(row=4, column=0, columnspan=2, sticky='nsew')
        self.stim_label = tk.Label(self.plot_data_frame, text="")
        self.stim_label.pack()

        self.plot_button = ttk.Button(self.load_navigate_frame, text='Plot Selected Index', command=self.update_plot)
        self.plot_button.grid(row=3, column=0, sticky='nsew')

        # #multiple index plot button
        # self.plot_multiple_button = ttk.Button(self.load_navigate_frame, text='Plot Multiple Indices', command=self.update_plot) 
        # self.plot_multiple_button.grid(row=3, column=1, sticky='nsew')

        #offset trace - make it a check button
        #create a check button to offset the trace
        # self.offset_button = ttk.Checkbutton(self.load_navigate_frame, text='Offset Trace', variable=self.offset_traces_pressed, command=self.offset_data)
        # self.offset_button = ttk.Button(self.load_navigate_frame, text='Offset Trace (make sure multple indices are empty)', command=self.offset_data)
        # self.offset_button.grid(row=3, column=2, sticky='nsew')

        # #average multiple indices button
        # self.average_multiple_button = ttk.Button(self.load_navigate_frame, text='Average Multiple Indices', command=self.average_multiple_plots)
        # self.average_multiple_button.grid(row=3, column=3, sticky='nsew')

    def widgets_export_frame(self):
        self.save_data_button = ttk.Button(self.export_frame, text='Save Data', command=self.save_data)
        self.save_data_button.pack(side=tk.TOP, pady=1)

        self.open_new_data_button = ttk.Button(self.export_frame, text='Open New Data', command=self.open_new_data)
        self.open_new_data_button.pack(side=tk.TOP, pady=1)

    def widgets_analysis_steps_frame(self):
        #add label for experiment preprocessing
        self.experiment_preprocessing_label = ttk.Label(self.analysis_steps_frame, text="Experiment Preprocessing", font=('Arial', 16, 'bold'))
        self.experiment_preprocessing_label.grid(row=0, column=0, sticky='nsew')

        #analysis_steps entry to be filled in with current_trace keys
        self.analysis_steps_label = ttk.Label(self.analysis_steps_frame, text="Analysis Steps", font=('Arial', 14, 'bold'))
        self.analysis_steps_label.grid(row=1, column=0, sticky='nsew')
        self.analysis_steps_entry = ttk.Entry(self.analysis_steps_frame, width=40)

        #add raw to the list
        self.analysis_steps_list.append('raw')
        self.analysis_steps_entry.insert(0, 'raw,')
        self.analysis_steps_entry.grid(row=1, column=1, sticky='nsew')

        #current_trace_key label dropdown 
        self.current_trace_key_label = ttk.Label(self.analysis_steps_frame, text="Current Trace Key", font=('Arial', 14, 'bold'))
        self.current_trace_key_label.grid(row=2, column=0, sticky='nsew')

        #current_trace_key dropdown menu
        self.current_trace_key_var = tk.StringVar(self.analysis_steps_frame)
        self.current_trace_key_dropdown = ttk.Combobox(self.analysis_steps_frame, textvariable=self.current_trace_key_var, values=['raw'], width=30)
        # insert the first value of the dropdown menu with the current_trace_key_var
        self.current_trace_key_dropdown.insert(0, 'raw')
        self.current_trace_key_dropdown.grid(row=2, column=1, sticky='nsew')

        '''Processing Steps for FI analysis and Cell Properties'''

        #detect spikes check button to plot spikes
        self.detect_spikes_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Detect Spikes", variable=self.detect_spikes_check, command=self.toggle_detect_spikes)
        self.detect_spikes_check_button.grid(row=3, column=0, sticky='nsew')

        #check button to analyze AP and AHP properties
        self.analyze_AP_AHP_properties_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Analyze AP Properties", variable=self.AP_AHP_analysis_check, command=self.toggle_AP_AHP_analysis)
        self.analyze_AP_AHP_properties_check_button.grid(row=4, column=0, sticky='nsew')

        #button to calculate input resistance and voltage sag
        self.calculate_intrinsic_properties_button = ttk.Button(self.analysis_steps_frame, text="Calculate Input Resistance and Voltage Sag", command=self.calculate_intrinsic_properties)
        self.calculate_intrinsic_properties_button.grid(row=5, column=0, sticky='nsew')

        '''Pre-procesing Steps for EPSP and Plateau Analysis'''

        #preprocessing label
        self.preprocessing_label = ttk.Label(self.analysis_steps_frame, text="Preprocessing Steps", font=('Arial', 14, 'bold'))
        self.preprocessing_label.grid(row=6, column=0, sticky='nsew')

        #add stim removal button check
        self.stim_removal_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Stim Removal - Please Select Experiment Type First and Fill Metadata",
                                                            variable=self.stim_removal_check, command=self.toggle_stim_removal)
        
        self.stim_removal_check_button.grid(row=7, column=0, sticky='nsew')

        #if it is pressed then add the stim_removal_pressed to True
        if self.stim_removal_check.get() == True:
            self.stim_removal_pressed = True

        #add noise removal check button
        self.noise_removal_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Noise Removal", variable=self.noise_removal_check, command=self.toggle_noise_removal)
        self.noise_removal_check_button.grid(row=8, column=0, sticky='nsew')

        if self.noise_removal_check.get() == True:
            self.noise_removal_pressed = True

        #add interpolated spikes check button
        self.interpolated_spikes_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Interpolated Spikes", variable=self.interpolated_spikes_check, command=self.toggle_interpolated_spikes)
        self.interpolated_spikes_check_button.grid(row=9, column=0, sticky='nsew')

        if self.interpolated_spikes_check.get() == True:
            self.noise_removal_pressed = True

        '''Post-processing Steps for EPSP and Plateau Analysis'''

        #post processing label
        self.post_processing_label = ttk.Label(self.analysis_steps_frame, text="Post Processing Steps", font=('Arial', 14, 'bold'))
        self.post_processing_label.grid(row=10, column=0, sticky='nsew')

        #partition_trace check button
        self.partition_traces_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Partition Trace", variable=self.partition_traces_check, command=self.toggle_partition_data)
        self.partition_traces_check_button.grid(row=11, column=0, sticky='nsew')

        if self.partition_traces_check.get() == True:
            self.partition_traces_pressed = True

        #add offset data check button
        self.offset_data_check_button = ttk.Checkbutton(self.analysis_steps_frame, text="Offset Data", variable=self.offset_traces_check, command=self.toggle_offset_data)
        self.offset_data_check_button.grid(row=12, column=0, sticky='nsew')

        if self.offset_traces_check.get() == True:
            self.offset_traces_pressed = True

        '''Labels and entries for pre and post processing steps'''

        #analyze AP properties title label
        self.AP_properties_label = ttk.Label(self.analysis_steps_frame, text="Analyze AP Properties", font=('Arial', 14, 'bold'))
        self.AP_properties_label.grid(row=13, column=0, sticky='nsew')

        self.Vm_rest_start_label = ttk.Label(self.analysis_steps_frame, text="Vm Rest Start (ms)")
        self.Vm_rest_start_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        self.Vm_rest_start_entry.insert(200, '200')
        #entry for Vm_rest_end
        self.Vm_rest_end_label = ttk.Label(self.analysis_steps_frame, text="Vm Rest End (ms)")
        self.Vm_rest_end_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        self.Vm_rest_end_entry.insert(300, '300')
        #entry for AP_peak_window_start
        self.AP_peak_window_start_label = ttk.Label(self.analysis_steps_frame, text="AP Peak Window Start (ms)")
        self.AP_peak_window_start_entry = ttk.Entry(self.analysis_steps_frame, width=10)    
        self.AP_peak_window_start_entry.insert(5, '5')

        #entry for AP_peak_window_end
        self.AP_peak_window_end_label = ttk.Label(self.analysis_steps_frame, text="AP Peak Window End (ms)")
        self.AP_peak_window_end_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        self.AP_peak_window_end_entry.insert(20, '20')

        #all the boxes to the GUI
        self.Vm_rest_start_label.grid(row=14, column=1, sticky='nsew')
        self.Vm_rest_start_entry.grid(row=14, column=0, sticky='nsew')

        self.Vm_rest_end_label.grid(row=15, column=1, sticky='nsew')
        self.Vm_rest_end_entry.grid(row=15, column=0, sticky='nsew')

        self.AP_peak_window_start_label.grid(row=16, column=1, sticky='nsew')
        self.AP_peak_window_start_entry.grid(row=16, column=0, sticky='nsew')

        #TODO add one to each row
        self.AP_peak_window_end_label.grid(row=17, column=1, sticky='nsew')
        self.AP_peak_window_end_entry.grid(row=17, column=0, sticky='nsew')

        #stim removal label
        self.stim_removal_label = ttk.Label(self.analysis_steps_frame, text="Stim Removal", font=('Arial', 14, 'bold'))
        self.stim_removal_label.grid(row=18, column=0, sticky='nsew')

        #entry for delete_start_stim
        self.delete_start_stim_label = ttk.Label(self.analysis_steps_frame, text="Delete Stim Artifacts Start (ms)")
        self.delete_start_stim_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        default_start_stim = self.defaults['filtering']['stim_removal']['start']
        self.delete_start_stim_entry.insert(0, f'{default_start_stim}')

        #entry for delete_end_stim
        self.delete_end_stim_label = ttk.Label(self.analysis_steps_frame, text="Delete Stim Artifacts End Duration (ms)")
        self.delete_end_stim_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        default_end_stim = self.defaults['filtering']['stim_removal']['end']
        self.delete_end_stim_entry.insert(0, f'{default_end_stim}')

        #all the boxes to the GUI
        self.delete_start_stim_label.grid(row=19, column=1, sticky='nsew')
        self.delete_start_stim_entry.grid(row=19, column=0, sticky='nsew')
        self.delete_end_stim_label.grid(row=20, column=1, sticky='nsew')
        self.delete_end_stim_entry.grid(row=20, column=0, sticky='nsew')

        #noise removal label
        self.noise_removal_label = ttk.Label(self.analysis_steps_frame, text="Noise Removal", font=('Arial', 14, 'bold'))
        self.noise_removal_label.grid(row=21, column=0, sticky='nsew')

        #entry for noise_times
        self.noise_times_label = ttk.Label(self.analysis_steps_frame, text="Noise Times (ms)")
        self.noise_times_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        #make it into a list if it is not already a list
        self.noise_start = self.defaults['filtering']['noise_removal']['start']
        self.noise_end = self.defaults['filtering']['noise_removal']['end']
        #make it into a list of start and end
        self.noise_times_entry.insert(0, f'{self.noise_start}, {self.noise_end}')
        #convert the noise times integer values
        self.noise_times = [int(time) for time in self.noise_times_entry.get().split(',')]
        self.noise_times_label.grid(row=22, column=1, sticky='nsew')
        self.noise_times_entry.grid(row=22, column=0, sticky='nsew')

        #interpolated spikes label
        self.interpolated_spikes_label = ttk.Label(self.analysis_steps_frame, text="Interpolated Spikes", font=('Arial', 14, 'bold'))
        self.interpolated_spikes_label.grid(row=23, column=0, sticky='nsew')

        #entry for spike_width
        self.spike_width_label = ttk.Label(self.analysis_steps_frame, text="Spike Width (ms)")
        self.spike_width_entry = ttk.Entry(self.analysis_steps_frame, width=10)
        default_spike_width = self.defaults['filtering']['interpolated_spikes']['spike_width']
        self.spike_width_entry.insert(0, f'{default_spike_width}')

        #entry for AP_peak_window_start
        self.AP_peak_window_start_label_interp = ttk.Label(self.analysis_steps_frame, text="AP Peak Window Start (ms)")
        self.AP_peak_window_start_entry_interp = ttk.Entry(self.analysis_steps_frame)
        #use the default value for the AP_peak_window_start
        self.AP_peak_window_start_entry_interp.insert(2, '2')

        #entry for AP_peak_window_end
        self.AP_peak_window_end_label_interp = ttk.Label(self.analysis_steps_frame, text="AP Peak Window End (ms)")
        self.AP_peak_window_end_entry_interp = ttk.Entry(self.analysis_steps_frame, width=10)
        #use the default value for the AP_peak_window_end
        self.AP_peak_window_end_entry_interp.insert(10, '10')

        #all the boxes to the GUI
        self.spike_width_label.grid(row=24, column=1, sticky='nsew')
        self.spike_width_entry.grid(row=24, column=0, sticky='nsew')

        self.AP_peak_window_start_label_interp.grid(row=25, column=1, sticky='nsew')
        self.AP_peak_window_start_entry_interp.grid(row=25, column=0, sticky='nsew')

        self.AP_peak_window_end_label_interp.grid(row=26, column=1, sticky='nsew')
        self.AP_peak_window_end_entry_interp.grid(row=26, column=0, sticky='nsew')

        #button to analyze EPSP peaks
        self.analyze_EPSP_peaks_button = ttk.Button(self.analysis_steps_frame, text="Analyze EPSP Peaks", command=self.analyze_EPSP_peaks)
        self.analyze_EPSP_peaks_button.grid(row=27, column=0, sticky='nsew')

    def widgets_experiment_metadata_frame(self):
        #add a label for the experiment metadata
        self.experiment_metadata_label = ttk.Label(self.experiment_metadata_frame, text="Experiment Metadata", font=('Arial', 16, 'bold'))
        self.experiment_metadata_label.grid(row=0, column=0, sticky='nsew')

        #test pulse
        self.test_pulse_label = ttk.Label(self.experiment_metadata_frame, text="Test Pulse")
        self.test_pulse_label.grid(row=1, column=0, sticky='nsew')

        self.test_pulse_start_label = ttk.Label(self.experiment_metadata_frame, text="Start (ms)")
        self.test_pulse_start_label.grid(row=2, column=1, sticky='nsew')
        self.test_pulse_start_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.test_pulse_start_entry.grid(row=2, column=0, sticky='nsew')

        self.test_pulse_end_label = ttk.Label(self.experiment_metadata_frame, text="End (ms)")
        self.test_pulse_end_label.grid(row=3, column=1, sticky='nsew')
        self.test_pulse_end_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.test_pulse_end_entry.grid(row=3, column=0, sticky='nsew')

        self.test_pulse_amp_label = ttk.Label(self.experiment_metadata_frame, text="Amplitude (pA)")
        self.test_pulse_amp_label.grid(row=4, column=1, sticky='nsew')
        self.test_pulse_amp_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.test_pulse_amp_entry.grid(row=4, column=0, sticky='nsew')

        #insert values for the test pulse convert values to integers
        self.test_pulse_start_entry.insert(0, f'{int(self.defaults["analysis"]["test_pulse"]["step_start"])}')
        self.test_pulse_end_entry.insert(0, f'{int(self.defaults["analysis"]["test_pulse"]["step_duration"])}')
        self.test_pulse_amp_entry.insert(0, f'{int(self.defaults["analysis"]["test_pulse"]["current_amplitude"])}')

        #label for filling in the experiment metadata
        self.experiment_metadata_fill_label = ttk.Label(self.experiment_metadata_frame, text="Fill in the Experiment Metadata")
        self.experiment_metadata_fill_label.grid(row=5, column=0, sticky='nsew')

        #buttons to fill in the metadata
        self.fill_metadata_button = ttk.Button(self.experiment_metadata_frame, text="Fill Metadata", command=self.fill_metadata)
        self.fill_metadata_button.grid(row=5, column=1, sticky='nsew')

        self.update_metadata_button = ttk.Button(self.experiment_metadata_frame, text="Update Metadata if Data is Missing", command=self.update_metadata)
        self.update_metadata_button.grid(row=5, column=2, sticky='nsew')

        #add a labels for the experiment metadata
        #pathway label in column 1
        self.pathway_label = ttk.Label(self.experiment_metadata_frame, text="Pathway")
        self.pathway_label.grid(row=6, column=0, sticky='nsew')

        #pathway entry in column 0
        self.pathway_entry_1 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.pathway_entry_1.grid(row=7, column=0, sticky='nsew')
        self.pathway_entry_2 = ttk.Entry(self.experiment_metadata_frame, width= 10)
        self.pathway_entry_2.grid(row=8, column=0, sticky='nsew')
        self.pathway_entry_3 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.pathway_entry_3.grid(row=9, column=0, sticky='nsew')
        self.pathway_entry_4 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.pathway_entry_4.grid(row=10, column=0, sticky='nsew')

        #Amplitude entries in column 1
        self.amplitude_label = ttk.Label(self.experiment_metadata_frame, text="Amplitude")
        self.amplitude_label.grid(row=6, column=1, sticky='nsew')

        self.amplitude_entry_1 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.amplitude_entry_1.grid(row=7, column=1, sticky='nsew')
        self.amplitude_entry_2 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.amplitude_entry_2.grid(row=8, column=1, sticky='nsew')
        self.amplitude_entry_3 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.amplitude_entry_3.grid(row=9, column=1, sticky='nsew')
        self.amplitude_entry_4 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.amplitude_entry_4.grid(row=10, column=1, sticky='nsew')

        #channel entries
        self.channel_label = ttk.Label(self.experiment_metadata_frame, text="Channel")
        self.channel_label.grid(row=6, column=2, sticky='nsew')

        self.channel_entry_1 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.channel_entry_1.grid(row=7, column=2, sticky='nsew')
        self.channel_entry_2 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.channel_entry_2.grid(row=8, column=2, sticky='nsew')
        self.channel_entry_3 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.channel_entry_3.grid(row=9, column=2, sticky='nsew')
        self.channel_entry_4 = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.channel_entry_4.grid(row=10, column=2, sticky='nsew')

        #ISI entry
        self.ISI_label = ttk.Label(self.experiment_metadata_frame, text="ISI (For E/I Experiments)")
        self.ISI_label.grid(row=11, column=0, sticky='nsew')
        self.ISI_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.ISI_entry.grid(row=11, column=1, sticky='nsew')

        #experiment type and description
        self.experiment_type_label = ttk.Label(self.experiment_metadata_frame, text="Experiment Type")
        self.experiment_type_label.grid(row=12, column=0, sticky='nsew')

        self.experiment_type_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.experiment_type_entry.grid(row=12, column=1, sticky='nsew')

        self.experiment_description_label = ttk.Label(self.experiment_metadata_frame, text="Experiment Description")
        self.experiment_description_label.grid(row=13, column=0, sticky='nsew')

        self.experiment_description_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.experiment_description_entry.grid(row=13, column=1, sticky='nsew')

        #Condition labels
        #make text Bolded
        self.condition_label = ttk.Label(self.experiment_metadata_frame, text="Condition") 
        self.condition_label.grid(row=14, column=0, sticky='nsew')
        #condition entry
        self.condition_entry = ttk.Entry(self.experiment_metadata_frame, width=10)
        self.condition_entry.grid(row=14, column=1, sticky='nsew')

        #metadata should be filled in with the default values
        self.fill_metadata()

    def widgets_plot_data_frame(self):
        '''Plot the data'''
        self.canvas = FigureCanvasTkAgg(figure.Figure(figsize=(10, 5)), master=self.plot_data_frame)
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_data_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (mV)')
    
    def widgets_plot_metadata_frame(self):
        # Navigation Buttons
        nav_button_frame = ttk.Frame(self.plot_metadata_frame) 
        nav_button_frame.grid(row=0, column=0, sticky='nsew')

        self.previous_button = ttk.Button(nav_button_frame, text="Previous", command=self.previous)
        self.previous_button.grid(row=0, column=0, sticky='nsew')

        self.next_button = ttk.Button(nav_button_frame, text="Next", command=self.next)
        self.next_button.grid(row=0, column=1, sticky='nsew')

        # Checkbox to clear the current plot
        self.clear_plot_check_button = ttk.Checkbutton(nav_button_frame, text="Clear Current Plot", variable=self.clear_plot)
        self.clear_plot_check_button.grid(row=0, column=2, sticky='nsew')

        # Checkbox to clear all plots
        self.clear_all_plots_check_button = ttk.Button(nav_button_frame, text="Clear All Plots", command=self.clear_all_plots)
        self.clear_all_plots_check_button.grid(row=0, column=3, sticky='nsew')

        # Create a frame to hold the title entry and label
        title_frame = ttk.Frame(self.plot_metadata_frame)
        title_frame.grid(row=1, column=0, sticky='nsew')

        self.plot_title_entry_label = ttk.Label(title_frame, text="Plot Title")
        self.plot_title_entry_label.grid(row=1, column=0, sticky='nsew')

        # Add an entry box for the title
        # default_title = 'Plotting Current Sweep'
        self.plot_title_entry = ttk.Entry(title_frame, width=20)
        # self.plot_title_entry.insert(0, default_title)
        self.plot_title_entry.grid(row=1, column=1, sticky='nsew')

        # Create a frame and entry for the raw data label on the plot
        raw_data_frame = ttk.Frame(self.plot_metadata_frame)
        raw_data_frame.grid(row=2, column=0, sticky='nsew')

        # Add the raw data plot check button
        self.raw_data_plot_check = ttk.Checkbutton(raw_data_frame, text='Plot Raw Data', variable=self.raw_data_plot_toggle, command=self.update_plot)
        self.raw_data_plot_check.grid(row=2, column=0, sticky='nsew')

        # Add the raw data label entry
        self.raw_data_label_entry = ttk.Entry(raw_data_frame, width=20)
        self.raw_data_label_entry.grid(row=2, column=1, sticky='nsew')

        # Insert the default value for the raw data label
        self.raw_data_label_entry.insert(0, 'Raw Data')

        # Add the raw label check button
        self.raw_label_check = ttk.Checkbutton(raw_data_frame, text='Label toggle', variable=self.raw_label_toggle, command=self.update_plot)
        self.raw_label_check.grid(row=2, column=2, sticky='nsew')

        # Add to the label legend list
        self.plot_labels_legend_list.append(self.raw_data_label_entry)

        # Create a frame for dynamically added elements
        self.dynamic_elements_frame = ttk.Frame(self.plot_metadata_frame)
        self.dynamic_elements_frame.grid(row=3, column=0, sticky='nsew')

        self.update_plot()

    def widgets_plot_r_inp_frame(self):
        self.canvas_r_inp = FigureCanvasTkAgg(figure.Figure(figsize=(10, 5)), master=self.plot_r_inp_frame)
        self.ax_r_inp = self.canvas_r_inp.figure.add_subplot(111)
        self.canvas_r_inp.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar_r_inp = NavigationToolbar2Tk(self.canvas_r_inp, self.plot_r_inp_frame)
        self.toolbar_r_inp.update()
        
        self.label_input_resistance = ttk.Label(self.plot_r_inp_frame, text="Calculate Input Resistance Across Experiments")
        self.label_input_resistance.pack(side=tk.TOP, pady=1)

        self.calculate_input_resistance_button = ttk.Button(self.plot_r_inp_frame, text='Plot Input Resistance', command=self.plot_input_resistance)
        self.calculate_input_resistance_button.pack(side=tk.TOP, pady=1)
    
    """Functions for handling single traces and general functions"""
    
    def update_metadata(self):
        #Update metadata for the current index
        #update discritpion and type
        self.experiment_description_entry.delete(0, tk.END)
        self.experiment_type_entry.delete(0, tk.END)
        self.data_df['experiment_description'].iloc[self.current_index] = self.experiment_description_entry.get()
        if self.experiment_type_entry.get() is not '':
            self.data_df['stim_type'].iloc[self.current_index] = self.experiment_type_entry.get()
        else:
            self.data_df['stim_type'].iloc[self.current_index] = ''

        #update the condition and ISI
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['condition'] = self.condition_entry.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['ISI'] = self.ISI_entry.get()

        #update the pathway entries
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_1_label'] = self.pathway_entry_1.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_2_label'] = self.pathway_entry_2.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_3_label'] = self.pathway_entry_3.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_4_label'] = self.pathway_entry_4.get()

        #update the amplitude entries
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_1_amp'] = self.amplitude_entry_1.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_2_amp'] = self.amplitude_entry_2.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_3_amp'] = self.amplitude_entry_3.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_4_amp'] = self.amplitude_entry_4.get()

        #update the channel entries
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_1_label'] = self.channel_entry_1.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_2_label'] = self.channel_entry_2.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_3_label'] = self.channel_entry_3.get()
        self.data_df['stimulus_metadata_dict'].iloc[self.current_index]['channel_4_label'] = self.channel_entry_4.get()

    def fill_metadata(self):
        # Fill in the metadata for the experiment
        self.update_description_and_type()
        self.change_text()
    
    def update_description_and_type(self):
        '''Update the experiment description and type when the index is changed using after()'''
        try:
            index = int(self.index_entry.get())
            if self.index_entry.get() != self.start_index:
                self.experiment_description_entry.delete(0, tk.END)
                self.experiment_description_entry.insert(0, self.data_df['experiment_description'].iloc[index])
                self.experiment_type_entry.delete(0, tk.END)
                self.experiment_type_entry.insert(0, self.data_df['stim_type'].iloc[index])
            # self.experiment_metadata_frame.after(100, self.update_description_and_type)
        except Exception as e:
            print(f"Error: {e}")

    def change_text(self):
        '''Change the text of all the labels in the GUI when the index is changed using after()'''
        try:
            if self.index_entry.get() != self.current_index:
                self.current_index = int(self.index_entry.get())
                #update multiple entries
                self.update_pathway_entries(self.current_index)
                self.update_amplitude_entries(self.current_index)
                #update single entries
                self.update_entry(self.ISI_entry, self.current_index, 'ISI')
                self.update_entry(self.condition_entry, self.current_index, 'condition')
            # self.experiment_metadata_frame.after(100, self.change_text)
        except Exception as e:
            print(f"Error: {e}")
    
    def update_entry(self, entry, index, key, suffix=''):
        '''Helper method to update a single entry widget.'''
        try:
            entry.delete(0, tk.END)
            metadata = self.data_df['stimulus_metadata_dict'].iloc[index]
            # debug
            # print(metadata, type(metadata))
            default_metadata = {'channel_1_amp': '', 'channel_2_amp': '', 'channel_3_amp': '', 'channel_4_amp': '',
                                'channel_1_label': '', 'channel_2_label': '', 'channel_3_label': '', 'channel_4_label': '',
                                'condition': '', 'ISI': ''}
            
            #check if the metadata is a dictionary and is empty
            if not isinstance(metadata, dict):
                metadata = default_metadata

            # Ensure the key is in the metadata dictionary
            if key not in metadata:
                metadata[key] = default_metadata[key]

            # Insert the value into the entry
            value = metadata[key]
            entry.insert(0, f'{(value)}{suffix}')

            #also add the value to the data_df for that index and key
            self.data_df['stimulus_metadata_dict'][index][key] = value

        except Exception as e:
            print(f"Error updating entry {key}: {e}")
    
    def update_pathway_entries(self, index):
        '''Helper method to update the pathway entries.'''
        pathway_keys = ['channel_1_label', 'channel_2_label', 'channel_3_label', 'channel_4_label']
        pathway_entries = [self.pathway_entry_1, self.pathway_entry_2, self.pathway_entry_3, self.pathway_entry_4]

        for key, entry in zip(pathway_keys, pathway_entries):
            self.update_entry(entry, index, key)

    def update_amplitude_entries(self, index):
        '''Helper method to update the amplitude entries.'''
        amplitude_keys = ['channel_1_amp', 'channel_2_amp', 'channel_3_amp', 'channel_4_amp']
        amplitude_entries = [self.amplitude_entry_1, self.amplitude_entry_2, self.amplitude_entry_3, self.amplitude_entry_4]

        for key, entry in zip(amplitude_keys, amplitude_entries):
            self.update_entry(entry, index, key)
        
    def open_file(self):
        try:
            self.file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
            if self.file_path:
                self.data_df = pd.read_pickle(self.file_path)
        
            return self.data_df
        
        except Exception as e:
            print(e)
            print('No file was selected.')

    def load_image(self, image_path):
        '''Load an image from a file path'''
        try:
            image = Image.open(image_path)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def get_plot_information(self, sweep):
        current_experiment = self.experiment_type_var.get()
        IV_stim_list = ['coarse_f_I_stim', 'fine_f_I_stim', 'voltage_sag']
        EPSP_stim_list = ['EPSP_test_pulses', 'E_I_pulse', 'theta_burst', 'tetanus']
        
        try:
            if current_experiment in IV_stim_list:
                stim_start = self.defaults['analysis'][current_experiment]['step_start']
                stim_end = self.defaults['analysis'][current_experiment]['step_duration'] + stim_start
                current_stim_array = self.data_df['stim_command'][sweep]
                stim_start_index = int(stim_start * self.data_df['acquisition_frequency'][sweep] / 1000)
                stim_end_index = int(stim_end * self.data_df['acquisition_frequency'][sweep] / 1000)
                current_stim_value = np.mean(current_stim_array[stim_start_index:stim_end_index])
                #round up to 4 decimal places
                current_stim_value = round(current_stim_value, 2)
                
            elif current_experiment in EPSP_stim_list:
                stim_start = np.nan
                stim_end = np.nan
                current_stim_value = np.nan

            else:
                stim_start = np.nan
                stim_end = np.nan
                current_stim_value = np.nan

            return stim_start, stim_end, current_stim_value
    
        except Exception as e:
            print(f"Error in get_plot_information: {e}")
            stim_start = np.nan
            stim_end = np.nan
            current_stim_value = np.nan
            return stim_start, stim_end, current_stim_value
        
    def apply_detect_spikes(self):
        try:
            index = self.current_index
            #function to detect spikes
            data = self.data_df['sweep'][index]
            data = data.copy()
            # TODO refer to yaml file for spike detection parameters
            self.AP_peaks_idx = find_peaks(data, height=-10)[0]

        except Exception as e:
            print("Error detecting spikes:", e)
    
    def plot_partitioned_and_offset_traces(self, traces, time, acquisition_frequency, label_toggle, additional_label):
        """
        Plot the offset traces for the different types of experiments.

        Parameters:
            traces (dict): A dictionary of the offset traces for the different channels.
            time (np.array): The time array for the current trace.
            acquisition_frequency (int): The acquisition frequency of the data.
            label_toggle (bool): A boolean to determine if the legend should be added to the plot.
            additional_label (str): An additional label to append to each trace label.
        """
        # for plotting ISI 300 experiments
        def plot_unitary_traces(add_legend):
            used_channels = set()
            for color, channel in zip(['black', 'red'], traces):
                baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                ISI_correction_time = self.defaults['analysis']['E_I_pulse'][channel]['ISI_correction']
                for stim in traces[channel]:
                    current_trace = traces[channel][stim][0]
                    current_time = np.linspace(stim - baseline_window_duration, stim + (self.ISI_time - ISI_correction_time), len(current_trace))
                    channel_label = f"{self.channel_names[channel]} {additional_label}" if channel not in used_channels and add_legend else None
                    self.ax.plot(current_time, current_trace, color=color, label=channel_label)
                    used_channels.add(channel)
            if add_legend:
                self.ax.legend()

        if self.experiment_type_var.get() in self.EPSP_list:
            if self.ISI_time == 300:
                plot_unitary_traces(add_legend=label_toggle)
            else:
                used_channels = set()
                for color, channel in zip(['black', 'red'], traces):
                    baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                    ISI_correction_time = self.defaults['analysis']['E_I_pulse'][channel]['ISI_correction']
                    current_trace = traces[channel][0]
                    stim_time = self.stim_start_dict[channel]
                    if self.num_pulses > 2:
                        current_time = np.linspace(stim_time - baseline_window_duration, stim_time + (self.ISI_time - ISI_correction_time), len(current_trace))
                        pathway_name = f"{self.channel_names[channel]} {additional_label}" if channel not in used_channels and label_toggle else None
                        used_channels.add(channel)
                        self.ax.plot(current_time, current_trace, color=color, label=pathway_name)
                    else:
                        stim_duration = int(15 * acquisition_frequency / 1000)
                        current_time = np.linspace(stim_time - baseline_window_duration, stim_time + stim_duration, len(current_trace))
                        pathway_name = f"{self.channel_names[channel]} {additional_label}" if channel not in used_channels and label_toggle else None
                        used_channels.add(channel)
                        self.ax.plot(current_time, current_trace, color=color, label=pathway_name)
                if label_toggle:
                    self.ax.legend()
        else:
            if label_toggle:
                self.ax.plot(time, traces, label=f'Offset Trace for {self.experiment_description_entry.get()} {additional_label}', color='k', linewidth=1)
                self.ax.legend()
            else:
                self.ax.plot(time, traces, color='k', linewidth=1)

    def update_plot(self):
        '''Update the plot based on the selected index.'''

        if self.index_entry.get() != '':
            index = int(self.index_entry.get())
        else:
            index = self.current_index

        if self.clear_plot.get():
            self.ax.clear()

        if self.raw_data_label_entry.get() != '':
            raw_label = self.raw_data_label_entry.get()
        else:
            raw_label = 'Raw Data'

        # Get the channel pathway names for the current index
        self.channel_names = {key: self.defaults['analysis']['E_I_pulse'][key]['label'] for key in self.defaults['analysis']['E_I_pulse']}

        default_title = f'Plotting Current Sweep {index}'
        if self.plot_title_entry.get() != '':
            title = self.plot_title_entry.get()
        else:
            title = default_title

        '''Plotting Raw Data'''

        data = self.data_df['sweep'][index]
        dt = self.data_df['dt'][index]
        self.time = np.arange(0, len(data) * dt, dt)

        # Make sure remove label is not empty and if check button is checked, plot with the label
        if self.raw_data_plot_toggle.get():
            if self.raw_label_toggle.get():
                self.ax.plot(self.time, data, color='grey', linewidth=1, label=raw_label, alpha=0.8)
                # Add the legend with this label
                self.ax.legend()
            else:
                self.ax.plot(self.time, data, color='grey', linewidth=1, alpha=0.8)

        '''Plotting AP analysis'''
        self.get_plot_information(index)

        stim_start, stim_end, stim_value = self.get_plot_information(index)

        if not np.isnan(stim_start) and not np.isnan(stim_end) and not np.isnan(stim_value):
            self.stim_label.config(text=f"IV Stim Start: {stim_start} ms, IV Stim End: {stim_end} ms, IV Stim Value: {stim_value} pA")
        else:
            self.stim_label.config(text="Stim information not available")

        if self.detect_spikes_check.get():
            AP_peaks_idx = self.AP_peaks_idx
            self.ax.scatter(self.time[AP_peaks_idx], data[AP_peaks_idx], marker='o', color='red', label='AP Peaks')
            self.ax.legend()

        if self.AP_AHP_analysis_check.get():
            AP_threshold_idx = self.AP_properties['AP threshold indices']
            AP_threshold = self.AP_properties['AP_threshold (mV)']
            AHP_idx = self.AP_properties['AHP_indices']

            self.ax.scatter(self.time[AP_threshold_idx], AP_threshold, marker='o', color='blue', label='AP Threshold')
            self.ax.scatter(self.time[AHP_idx], data[AHP_idx], marker='o', color='green', label='AHP')

            self.ax.legend()
        # Iterate over all the traces that it finds in current_trace keys for intermediate traces
        trace_keys_list = list(self.data_df['intermediate_traces'][index].keys())
        self.current_row = 0
        for trace_key in trace_keys_list:
            # Make sure it only does it once
            if trace_key not in self.plot_check_buttons.keys():
                # Generate all the GUI elements needed for plot checkboxes and label textboxes for this trace_key
                # Create the new frame for the elements
                new_elements_frame = ttk.Frame(self.dynamic_elements_frame)
                new_elements_frame.pack(side=tk.TOP, fill=tk.X, pady=2)

                # Add the check button boolean for that trace key to the analysis_check_buttons dictionary
                self.plotting_checks[trace_key] = tk.BooleanVar(value=False)
                #capitilize the first letter of the trace key for the plot text
                plot_text = trace_key.capitalize()
                check_button = ttk.Checkbutton(new_elements_frame, text=plot_text, variable=self.plotting_checks[trace_key], command=self.update_plot)
                check_button.pack(side=tk.LEFT, padx=5, pady=2)
                self.plot_check_buttons[trace_key] = check_button

                # Create an entry for the processed data label
                stim_data_label = ttk.Entry(new_elements_frame, width=15)
                stim_data_label.pack(side=tk.LEFT, padx=5, pady=2)
                stim_data_label.insert(0, f'{trace_key}')
                self.processed_data_label[trace_key] = stim_data_label

                # Pack a check button for the processed data label
                self.label_checks[trace_key] = tk.BooleanVar(value=False)
                label_check = ttk.Checkbutton(new_elements_frame, text='Label toggle', variable=self.label_checks[trace_key], command=self.update_plot)
                label_check.pack(side=tk.LEFT, padx=5, pady=2)
                self.label_check_buttons[trace_key] = label_check

                # Add the label to the legend list
                self.plot_labels_legend_list.append(stim_data_label)

                # Add the new widgets to the plot_widgets dictionary
                if trace_key not in self.plot_widgets:
                    self.plot_widgets[trace_key] = []
                self.plot_widgets[trace_key].extend([new_elements_frame, stim_data_label, label_check, check_button])

                self.current_row += 1
            
            #if the trace key does not exist then destroy it from the new_elements_frame
            if trace_key not in self.data_df['intermediate_traces'][index]:
                for widget in self.plot_widgets[trace_key]:
                    widget.destroy()
                    
                    self.plot_widgets.clear()
                    self.plot_check_buttons.clear()
                    self.processed_data_label.clear()
                    self.label_check_buttons.clear()
                
        # Plotting selected intermediate traces
        for trace_key in trace_keys_list:
            if self.plotting_checks[trace_key].get():
                label_toggle = self.label_checks[trace_key].get()
                if trace_key == 'partitioned_trace' or trace_key == 'offset_trace':
                    traces = self.data_df['intermediate_traces'][index][trace_key]
                    if label_toggle:
                        processed_label = self.processed_data_label[trace_key].get()
                        self.plot_partitioned_and_offset_traces(traces, self.time, self.data_df['acquisition_frequency'][index], label_toggle=label_toggle, additional_label=processed_label)
                    else:
                        self.plot_partitioned_and_offset_traces(traces, self.time, self.data_df['acquisition_frequency'][index], label_toggle=label_toggle, additional_label='')
                else:
                    processed_data = self.data_df['intermediate_traces'][index][trace_key]
                    if label_toggle:
                        processed_label = self.processed_data_label[trace_key].get()
                        self.ax.plot(self.time, processed_data, color='blue', linewidth=1, label=processed_label)
                        self.ax.legend()
                    else:
                        self.ax.plot(self.time, processed_data, color='blue', linewidth=1)

        self.ax.set_title(title)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Voltage (mV)')
        self.canvas.draw()

    def average_multiple_plots(self):
        #make an instance for if the traces have been offset already            
        #average the multiple plots if they are not offset
        self.average_traces_pressed = True
        if ',' in self.index_entry_multiple.get():
            temp_plot_indices = self.index_entry_multiple.get().split(',')
            temp_plot_indices = [int(index) for index in temp_plot_indices]

            temp_traces = []
            for i in temp_plot_indices:
                temp_traces.append(self.data_df['sweep'][i])
            
            temp_plot_traces = []
            for trace in temp_traces:
                temp_plot_traces.append(trace[:len(temp_traces[0])])
            
            if self.offset_traces == {}:
                self.average_data = np.mean(temp_plot_traces, axis=0)
            
            elif self.offset_traces != {}:
                offset_traces = self.offset_traces
                if self.ISI_time == 300:
                    #create a temp dictionary of each average trace per channel at each stim time and for each index
                    self.average_offset_traces = {}
                    for index in offset_traces:
                        self.average_offset_traces[index] = {}
                        for channel in offset_traces[index]:
                            self.average_offset_traces[index][channel] = {}
                            for stim in offset_traces[index][channel]:
                                self.average_offset_traces[index][channel][stim] = np.mean(offset_traces[index][channel][stim], axis=0)
                else:
                    self.average_offset_traces = {}
                    for index in offset_traces:
                        self.average_offset_traces[index] = {}
                        for channel in offset_traces[index]:
                            self.average_offset_traces[index][channel] = np.mean(offset_traces[index][channel], axis=0)
            
            else:
                self.average_data = np.mean(temp_plot_traces, axis=0)
    
    def apply_partition_data(self):
        '''partition the data based on the experiment type'''

        index = self.current_index
        acquisition_frequency = self.data_df['acquisition_frequency'][self.start_index]

        if self.data_df['current_trace_key'][index] == 'raw':
                self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]
        
        acquisition_frequency = self.data_df['acquisition_frequency'][self.start_index]
        self.stim_start_dict = {key: self.defaults['analysis'][self.experiment_type_var.get()][key]['stim_start'] for key in self.defaults['analysis'][self.experiment_type_var.get()]}

        if self.experiment_type_var.get() in self.EPSP_list:
            self.partition_data_dict = {key: [] for key in self.defaults['analysis']['E_I_pulse']}

            #check how many pulses are in the experiment for the first channel (key)
            self.num_pulses = self.defaults['analysis'][self.experiment_type_var.get()][list(self.defaults['analysis'][self.experiment_type_var.get()].keys())[0]]['num_pulses']

            #if the pulses 3 or more, then it is an E/I experiment with and ISI time
            if self.num_pulses > 2:
                ISI_time = int(float(self.ISI_entry.get()))
                self.ISI_time = ISI_time

            #handle unitary EPSPs
            if self.ISI_time == 300:
                for channel in self.partition_data_dict:
                    current_stim_times_channel = []
                    stim_start_time_channel = self.stim_start_dict[channel]

                    ISI_correction_time = self.defaults['analysis']['E_I_pulse'][channel]['ISI_correction']

                    for pulse in range(self.defaults['analysis']['E_I_pulse'][channel]['num_pulses']):
                        current_stim_times_channel.append(stim_start_time_channel + (pulse * (self.ISI_time - ISI_correction_time)))
                    
                    self.partition_data_dict[channel] = {key: [] for key in current_stim_times_channel}

                for channel in self.partition_data_dict:
                        baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                        baseline_window_duration = int(baseline_window_duration * acquisition_frequency / 1000)
                        for stim_time in self.partition_data_dict[channel]:
                            start_stim_index = int(stim_time * acquisition_frequency / 1000)
                            #start 20ms before the stim time 
                            partition_start_index = start_stim_index - baseline_window_duration
                            partition_end_index = start_stim_index + int((self.ISI_time - ISI_correction_time) * acquisition_frequency / 1000)
                            self.partition_data_dict[channel][stim_time].append(self.current_trace[partition_start_index:partition_end_index])
            
            #for other experiments E/I not unitary EPSPs there is just a channel key
            else:
                for i, channel in enumerate(self.partition_data_dict):
                    baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                    baseline_window_duration = int(baseline_window_duration * acquisition_frequency / 1000)
                    stim_time = self.stim_start_dict[channel]
                    stim_index = int(stim_time * acquisition_frequency / 1000)
                    #take 10ms before the stim time
                    partition_start_index = stim_index - baseline_window_duration
                    #get a trace that is 300ms from the stim start time
                    self.partition_data_dict[channel].append(self.current_trace[partition_start_index: stim_index + int(300 * acquisition_frequency / 1000)])
                    self.current_trace = self.partition_data_dict
    
        #update the intermediate trace for the current index
        self.data_df['intermediate_traces'][self.current_index]['partitioned_trace'] = self.partition_data_dict
        self.data_df['current_trace_key'].at[index] = 'partitioned_trace'

    def toggle_partition_data(self):
        if self.partition_traces_check.get():
            self.analysis_steps_list.append('partitioned_trace')
            self.analysis_steps_entry.insert(tk.END, 'partitioned_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to partitioned_trace
            self.current_trace_key_dropdown.set('partitioned_trace')
        else:
            if 'partitioned_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('partitioned_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot()
    
    def apply_offset_data(self):
        '''Offset the data by subtracting the mean of a window before the stimulus'''
        index = self.current_index
        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]
            acquisition_frequency = self.data_df['acquisition_frequency'][self.start_index]

        self.stim_start_dict = {key: self.defaults['analysis'][self.experiment_type_var.get()][key]['stim_start'] for key in self.defaults['analysis'][self.experiment_type_var.get()]}

        if self.experiment_type_var.get() in self.EPSP_list:
            self.offset_data_dict = {key: [] for key in self.defaults['analysis']['E_I_pulse']}
            self.num_pulses = self.defaults['analysis'][self.experiment_type_var.get()][list(self.defaults['analysis'][self.experiment_type_var.get()].keys())[0]]['num_pulses']

            if self.num_pulses > 2:
                ISI_time = int(float(self.ISI_entry.get()))
                self.ISI_time = ISI_time

            if self.ISI_time == 300:
                for channel in self.offset_data_dict:
                    current_stim_times_channel = []
                    stim_start_time_channel = self.stim_start_dict[channel]
                    ISI_correction_time = self.defaults['analysis']['E_I_pulse'][channel]['ISI_correction']

                    for pulse in range(self.defaults['analysis']['E_I_pulse'][channel]['num_pulses']):
                        current_stim_times_channel.append(stim_start_time_channel + (pulse * (self.ISI_time - ISI_correction_time)))
                    
                    self.offset_data_dict[channel] = {key: [] for key in current_stim_times_channel}

                for channel in self.partition_data_dict:
                    baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                    for stim_time in self.partition_data_dict[channel]:
                        current_trace = self.partition_data_dict[channel][stim_time][0]
                        offset_window_samples = int(baseline_window_duration * acquisition_frequency / 1000)  
                        # Ensure the start index is non-negative and the window is within trace length
                        start_index = 0
                        end_index = min(offset_window_samples, len(current_trace))
                        # Get the offset window trace
                        offset_window_trace = current_trace[start_index:end_index]
                        # Calculate the mean of the offset window
                        mean_offset_value = np.mean(offset_window_trace)
                        # Subtract the mean from the current trace to offset it
                        offset_corrected_trace = current_trace - mean_offset_value
                        self.offset_data_dict[channel][stim_time].append(offset_corrected_trace)

                self.offset_traces = self.offset_data_dict
                self.current_trace = self.offset_traces                    
            #for other experiments E/I not unitary EPSPs there is just a channel key
            else:
                for channel in self.partition_data_dict:
                    baseline_window_duration = self.defaults['analysis']['E_I_pulse'][channel]['baseline_window_duration']
                    current_trace = self.partition_data_dict[channel][0]
                    offset_window_samples = int(baseline_window_duration * acquisition_frequency / 1000)  # 15 milliseconds offset
                    start_index = 0
                    end_index = min(offset_window_samples, len(current_trace))
                    offset_window_trace = current_trace[start_index:end_index]
                    mean_offset_value = np.mean(offset_window_trace)
                    offset_corrected_trace = current_trace - mean_offset_value
                    self.offset_data_dict[channel].append(offset_corrected_trace)

                self.offset_traces = self.offset_data_dict
                self.current_trace = self.offset_traces
        
        #for other experiments such as plateau experiments
        if self.experiment_type_var.get() in self.plateau_list:
            current_trace = self.current_trace
            stim_start = self.stim_start_dict[list(self.stim_start_dict.keys())[0]]
            stim_start_index = int(stim_start * acquisition_frequency / 1000)
            offset_window = current_trace[stim_start_index - int(10 * acquisition_frequency / 1000): stim_start_index]
            offset_value = np.mean(offset_window)
            offset_trace = current_trace - offset_value
            self.offset_traces = offset_trace
            self.current_trace = self.offset_traces
        
        #update the intermediate trace for the current index
        self.data_df['intermediate_traces'][self.current_index]['offset_trace'] = self.current_trace
        self.data_df['current_trace_key'].at[index] = 'offset_trace'
    
    def toggle_offset_data(self):
        if self.offset_traces_check.get():
            self.analysis_steps_list.append('offset_trace')
            self.analysis_steps_entry.insert(tk.END, 'offset_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('offset_trace')
        else:
            if 'offset_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('offset_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot()

    def clear_all_plots(self):
        '''Clear all plots on the canvas'''
        self.ax.clear()
        self.canvas.draw()
    
    def next(self):
        '''Move to the next index and update the plot'''
        self.current_index = int(self.index_entry.get())
        #should also refill the metadata entries
        if self.current_index < len(self.data_df) - 1:
            self.fill_metadata()
            self.index_entry.delete(0, tk.END)
            self.index_entry.insert(0, str(self.current_index + 1))
            self.current_index += 1
            self.apply_analysis_steps()
            self.update_plot()
        
    def previous(self):
        '''Move to the previous index and update the plot'''
        self.current_index = int(self.index_entry.get())
        if self.current_index > 0:
            self.fill_metadata()
            self.index_entry.delete(0, tk.END)
            self.index_entry.insert(0, str(self.current_index - 1))
            self.current_index -= 1
            self.apply_analysis_steps()
            self.update_plot()
    
    def apply_analysis_steps(self):
        '''Check the toggle states and run the appropriate functions'''

        self.apply_detect_spikes()
        self.apply_AP_AHP_analysis()

        for trace_key in self.analysis_steps_list:
            if trace_key == 'raw':
                self.data_df['current_trace_key'].at[self.current_index] = trace_key
            elif trace_key == 'stim_removed_trace':
                if 'stim_removed_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_stim_removal()
                else:
                    #TODO: # wrapper function that does this and changes the current trace key pulldown
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key        
            elif trace_key == 'noise_removed_trace':
                if 'noise_removed_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_noise_removal()
                else:
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key
            elif trace_key == 'interpolated_spikes_trace':
                if 'interpolated_spikes_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_interpolated_spikes()
                else:
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key
            elif trace_key == 'partitioned_trace':
                if 'partitioned_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_partition_data()
                else:
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key
            elif trace_key == 'offset_trace':
                if 'offset_trace' not in self.data_df['intermediate_traces'][self.current_index]:
                    self.apply_offset_data()
                else:
                    self.data_df['current_trace_key'].at[self.current_index] = trace_key
    
    
    # #Get the input resistance across every sweep
    def get_input_resistances(self):
        #Get the input resistance across every sweep
        acquisition_frequency = self.data_df['acquisition_frequency'][self.start_index]
        current_pulse_amp = self.test_pulse_amp_entry.get()
        start_time = self.test_pulse_start_entry.get()
        end_time = self.test_pulse_end_entry.get()
        #have running list of input resistance values
        input_resistances = []

        all_sweeps = self.data_df['sweep'][:]
        #Get the array from all the sweeps and store them in a list of traces
        traces = []
        for sweep in all_sweeps:
            traces.append(sweep)

        for trace in traces:
            input_resistance = calculate_input_resistance(trace, current_pulse_amp, acquisition_frequency, start_time, end_time)
            input_resistances.append(input_resistance)
        
        return input_resistances
    
    # #Plot the input resistance across all the sweeps using experiment time
    def plot_input_resistance(self):
        input_resistances = self.get_input_resistances()

        #fill in the data frame of the average input resistance
        # self.analysis_types['Intrinsic_cell']['steady_state_input_resistance'] = np.mean(input_resistances)
        #create the time array
        time_points = []
        #enumerate through all the sweeps
        for i in range(len(self.data_df)):
            first_time = self.data_df['global_wall_clock'][0]
            time_point = (self.data_df['global_wall_clock'][i] - first_time)/60
            time_points.append(time_point)

        #now plot the input resistance for each experiment against the cumulative time points 
        self.ax_r_inp.scatter(time_points, input_resistances, color='black', linewidth=1, label='Input Resistance')
        self.ax_r_inp.set_xlabel('Time (min)')
        self.ax_r_inp.set_ylabel('Input Resistance (MΩ)')
        self.ax_r_inp.set_title('Input Resistance Across Experiments')
        self.ax_r_inp.legend()
        self.canvas_r_inp.draw()

    "All Toggles"

    def toggle_detect_spikes(self):
        if self.detect_spikes_check.get():
            self.apply_analysis_steps()
            self.update_plot()

    # #Toggle analyze AP AHP properties to plot the AP threshold and AHP
    def apply_AP_AHP_analysis(self):
        if self.index_entry.get() != '':
            index = int(self.index_entry.get())
        else:
            index = self.start_index
        #run the AP AHP analysis function
        temp_AP_peak_idx = self.AP_peaks_idx
        time = np.arange(0, len(self.data_df['sweep'][index]) * self.data_df['dt'][index], self.data_df['dt'][index])
        Vm_rest_start = int(self.Vm_rest_start_entry.get())
        Vm_rest_end = int(self.Vm_rest_end_entry.get())
        AP_peak_window_start = int(self.AP_peak_window_start_entry.get())
        AP_peak_window_end = int(self.AP_peak_window_end_entry.get())
        stim_start, stim_end, stim_value = self.get_plot_information(index)
        # # trace, time, AP_peak_idx, stim_length, acquisition_frequency, Vm_rest_start, Vm_rest_end, AP_peak_window_start, AP_peak_window_end)
        self.AP_properties = analyze_AP_AHP_properties_trace(self.data_df['sweep'][index], time, temp_AP_peak_idx, stim_end, stim_end-stim_start, self.data_df['acquisition_frequency'][index], 
                                                             Vm_rest_start, Vm_rest_end, AP_peak_window_start, AP_peak_window_end)
        self.analysis_types['AP'] = {key:None for key in self.analysis_types['AP']}
        self.analysis_types['AP']['AP_threshold'] = self.AP_properties['AP_threshold (mV)'][:]
        self.analysis_types['AP']['AP_threshold_indices'] = self.AP_properties['AP threshold indices'][:]
        self.analysis_types['AP']['Vm_rest'] = self.AP_properties['Vm_rest (mV)']
        self.analysis_types['AP']['AP_firing_rate'] = self.AP_properties['AP_firing_rate (Hz)']
        self.analysis_types['AP']['AP_number'] = self.AP_properties['AP_number']
        self.analysis_types['AP']['AP_ISI_time'] = self.AP_properties['AP_ISI_time (ms)'][:]
        self.analysis_types['AP']['AHP_indices'] = self.AP_properties['AHP_indices'][:]
        self.analysis_types['AP']['AP_halfwidth'] = self.AP_properties['AP_halfwidth (ms)'][:]
        self.analysis_types['AP']['AP_size'] = self.AP_properties['AP_size (mV)'][:]
        self.analysis_types['AP']['AHP_halfwidth'] = self.AP_properties['AHP_halfwidth (ms)'][:]
        self.analysis_types['AP']['AHP_size'] = self.AP_properties['AHP_size (mV)'][:]

        #get the current amplitude and firing rate for FI plot
        current_stim_value = self.get_plot_information(index)[2]
        experiment_type = self.experiment_type_var.get()

        self.analysis_types[experiment_type] = {key:None for key in self.analysis_types[experiment_type]}
        self.analysis_types[experiment_type]['current_amplitudes'] = current_stim_value
        self.analysis_types[experiment_type]['firing_rates'] = self.AP_properties['AP_firing_rate (Hz)']

        #save this dictionary to the data_df of analysis_dict
        self.data_df['analysis_dict'].at[index] = self.analysis_types
        
    def toggle_AP_AHP_analysis(self):
        if self.AP_AHP_analysis_check.get():
            self.apply_analysis_steps()
            self.update_plot()
    
    def calculate_intrinsic_properties(self):
        #calculate the intrinsic properties of the cell including the input resistance, voltage sag...
        if self.index_entry.get() != '':
            index = int(self.index_entry.get())
        
        experiment_type = self.experiment_type_var.get()
        
        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]
        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]

        #get the input resistance
        input_resistances = self.get_input_resistances()
        average_input_resistance = np.mean(input_resistances, axis=0)

        self.analysis_types['Intrinsic_cell'] = {key:None for key in self.analysis_types['Intrinsic_cell']}
        self.analysis_types['Intrinsic_cell']['steady_state_input_resistance'] = average_input_resistance

        #calculate the voltage sag
        start_time = self.defaults['analysis']['voltage_sag']['step_start']
        end_time = self.defaults['analysis']['voltage_sag']['step_duration'] + start_time
        # print(f'Start Time: {start_time}, End Time: {end_time}')
        voltage_sag = calculate_voltage_sag(experiment_type, self.current_trace, self.data_df['acquisition_frequency'][index], start_time, end_time)
        self.analysis_types['Intrinsic_cell']['Voltage_sag'] = voltage_sag

        self.data_df['analysis_dict'][index] = self.analysis_types
        
        print(f'Average Input Resistance: {average_input_resistance} MΩ')
        print(f'Voltage Sag: {voltage_sag} %')
    
    def analyze_EPSP_peaks(self):
        #get the EPSP peaks - for this will take in dictionary of traces that have been partitioned and offset
        #for a single trace it will find peaks for each partitioned trace and for multiple traces it will find peaks for each partitioned trace of the average trace

        #analyze the EPSP peaks
        acquisition_frequency = self.data_df['acquisition_frequency'][self.current_index]
        delete_start = self.defaults['filtering']['stim_removal']['start']
        delete_start_index = int(delete_start * acquisition_frequency/1000)

        traces = self.offset_data_dict
        window_size = 100

        #get the EPSP peaks
        if self.ISI_time == 300:
                for channel in traces:
                    for stim in traces[channel]:
                        current_trace = traces[channel][stim][0]
                        current_time = np.linspace(stim, stim + self.ISI_time, len(current_trace))
                        EPSP_peak, _ = find_peaks_in_window(current_trace, delete_start_index, window_size, acquisition_frequency)

                        self.analysis_types['E_I_pulse'] = {key:None for key in self.analysis_types['E_I_pulse']}
                        self.analysis_types['E_I_pulse']['EPSP_amplitude'] = EPSP_peak
                        print(f'EPSP Peak Size in mV: {EPSP_peak}')

                        EPSP_peak_time = current_time[EPSP_peak]

                        #plot the peaks
                        self.ax.scatter(EPSP_peak_time, EPSP_peak, marker='o', color='red', label='EPSP Peaks')

        else:
                for channel in traces:
                    current_trace = traces[channel][0]
                    stim_start = self.stim_start_dict[channel]
                    
                    if self.num_pulses > 2:
                        current_time = np.linspace(stim_start, stim_start + self.ISI_time, len(current_trace))
                        EPSP_peak, _ = find_peaks_in_window(current_trace, delete_start_index, window_size, acquisition_frequency)
                        self.analysis_types['E_I_pulse'] = {key:None for key in self.analysis_types['E_I_pulse']}
                        self.analysis_types['E_I_pulse']['EPSP_amplitude'] = EPSP_peak
                        print(f'EPSP Peak Size in mV: {EPSP_peak}')
                        EPSP_peak_time = current_time[EPSP_peak]

                        #plot the peaks
                        self.ax.scatter(EPSP_peak_time, EPSP_peak, marker='o', color='red', label='EPSP Peaks')
                    else: 
                        stim_duration = 15
                        current_time = np.linspace(stim_start, stim_start + stim_duration, len(current_trace))
                        EPSP_peak, EPSP_peak_idx = find_peaks_in_window(current_trace, delete_start_index, window_size, acquisition_frequency)
                        self.analysis_types['E_I_pulse'] = {key:None for key in self.analysis_types['E_I_pulse']}
                        self.analysis_types['E_I_pulse']['EPSP_amplitude'] = EPSP_peak
                        print(f'EPSP Peak Size in mV: {EPSP_peak}')

                        EPSP_peak_time = current_time[EPSP_peak_idx]

                        #plot the peaks
                        self.ax.scatter(EPSP_peak_time, EPSP_peak, marker='o', color='red', label='EPSP Peaks')
        
        self.canvas.draw()

    def get_stim_times(self):
        experiment_type = self.experiment_type_var.get()
        non_E_I_pulse_stims = ['EPSP_test_pulses', 'theta_burst', 'tetanus']
        self.stim_times = []
        if experiment_type in non_E_I_pulse_stims:
            for channel in self.defaults['analysis'][experiment_type]:
                stim_start = self.defaults['analysis'][experiment_type][channel]['stim_start']
                for i in range(self.defaults['analysis'][experiment_type][channel]['num_pulses']):
                    self.stim_times.append(stim_start + i*self.defaults['analysis'][experiment_type][channel]['ISI'])

        if experiment_type == 'E_I_pulse':
            ISI_time = self.data_df['stimulus_metadata_dict'][self.current_index]['ISI']
            if ISI_time is not None:
                ISI_time = int(ISI_time)

            self.stim_times = []
            for channel in self.defaults['analysis'][experiment_type]:
                stim_start = self.defaults['analysis'][experiment_type][channel]['stim_start']
                for i in range(self.defaults['analysis'][experiment_type][channel]['num_pulses']):
                    if ISI_time in self.defaults['analysis'][experiment_type][channel]['ISI_list']:
                        self.stim_times.append(stim_start + i*ISI_time)  

        return self.stim_times

    def apply_stim_removal(self):   
        index = self.current_index

        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]

        if self.experiment_type_var.get() in self.EPSP_list:
                stim_times = self.get_stim_times()
                temp_data = remove_artifacts_custom(self.current_trace, stim_times, self.data_df['acquisition_frequency'][index], float(self.delete_start_stim_entry.get()), float(self.delete_end_stim_entry.get()))
                self.current_trace = temp_data

        if self.experiment_type_var.get() in self.plateau_list:
            time = np.arange(0, len(self.current_trace) * self.data_df['dt'][index], self.data_df['dt'][index])
            temp_data = remove_artifacts_plateau(self.current_trace, time, acquisition_frequency=self.data_df['acquisition_frequency'][index], 
                                                delete_start_stim=float(self.delete_start_stim_entry.get()), delete_end_stim=float(self.delete_end_stim_entry.get()), height=0)
            self.current_trace = temp_data

        #insert into the intermediate traces dictionary of the data frame for the current index
        self.data_df['intermediate_traces'][index]['stim_removed_trace'] = self.current_trace
        self.data_df['current_trace_key'].at[index] = 'stim_removed_trace'

    def toggle_stim_removal(self):
        if self.stim_removal_check.get():
            self.analysis_steps_list.append('stim_removed_trace')
            self.analysis_steps_entry.insert(tk.END, 'stim_removed_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('stim_removed_trace')
        else:
            if 'stim_removed_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('stim_removed_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot() 
    
    def apply_noise_removal(self):
        index = self.current_index
        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]
        noise_times = self.noise_times
        temp_data = remove_artifacts_custom(self.current_trace, noise_times, self.data_df['acquisition_frequency'][index], float(self.delete_start_stim_entry.get()), float(self.delete_end_stim_entry.get()))
        self.data_df['intermediate_traces'][index]['noise_removed_trace'] = temp_data
        self.current_trace = temp_data
        
        self.data_df['intermediate_traces'][index]['noise_removed_trace'] = self.current_trace
        self.data_df['current_trace_key'].at[index] = 'noise_removed_trace'

    def toggle_noise_removal(self):
        if self.noise_removal_check.get():
            self.analysis_steps_list.append('noise_removed_trace')
            self.analysis_steps_entry.insert(tk.END, 'noise_removed_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('noise_removed_trace')
        
        else:
            if 'noise_removed_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('noise_removed_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot() 
    
    def apply_interpolated_spikes(self):
        index = self.current_index

        if self.data_df['current_trace_key'][index] == 'raw':
            self.current_trace = self.data_df['sweep'][index]

        else:
            self.current_trace = self.data_df['intermediate_traces'][index][self.data_df['current_trace_key'][index]]

        dt = self.data_df['dt'][index]
        acquisition_frequency = self.data_df['acquisition_frequency'][index]
        time = np.arange(0, len(self.current_trace) * dt, dt)
        spike_width = float(self.spike_width_entry.get())
        AP_peak_window_start_interp = float(self.AP_peak_window_start_entry_interp.get())
        AP_peak_window_end_interp = float(self.AP_peak_window_end_entry_interp.get())
        temp_data = interp_spikes(AP_peak_window_start_interp, AP_peak_window_end_interp, spike_width, acquisition_frequency, self.current_trace, time, height=self.defaults['filtering']['interpolated_spikes']['threshold'])
        self.current_trace = temp_data

        self.data_df['intermediate_traces'][index]['interpolated_spikes_trace'] = self.current_trace
        self.data_df['current_trace_key'].at[index] = 'interpolated_spikes_trace'

    def toggle_interpolated_spikes(self):
        if self.interpolated_spikes_check.get():
            self.analysis_steps_list.append('interpolated_spikes_trace')
            self.analysis_steps_entry.insert(tk.END, 'interpolated_spikes_trace,')
            self.current_trace_key_dropdown['values'] = self.analysis_steps_list
            #set the current trace key in the dropdown to offset_trace
            self.current_trace_key_dropdown.set('interpolated_spikes_trace')
        
        else:
            if 'interpolated_spikes_trace' in self.analysis_steps_list:
                self.analysis_steps_list.remove('interpolated_spikes_trace')
                self.analysis_steps_entry.delete(0, tk.END)
                for element in self.analysis_steps_list:
                    self.analysis_steps_entry.insert(tk.END, f'{element},')

        self.apply_analysis_steps()
        self.update_plot() 

    #Save the data to a pickle file
    def save_data(self):
        try:
            data_pickle_name = simpledialog.askstring("Data Pickle File", "Enter a name (date_cell):") 
            save_path = simpledialog.askstring("Save Path", "Enter a save path (e.g. /content/drive/My Drive/):")
            if data_pickle_name is not None and save_path is not None:
                data_df = self.data_df  
                data_df.to_pickle(f'{save_path}/{data_pickle_name}.pkl')
        except Exception as e:
            print("Error saving data:", e)
    
    def open_new_data(self):
        try:
            self.data_df = self.open_file()
            self.fill_metadata()
        except Exception as e:
            print(f"Error opening new data: {e}")