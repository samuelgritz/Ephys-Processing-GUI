# Electrophysiology Processing GUI 

A work in progress GUI that takes in .pkl files converted from .pxp files generated by NeuroMatic, an Igor Pro software package for acquisition, analysis and simulation of electrophysiological data. 

Processed file example used for demonstration: [https://drive.google.com/file/d/1awD2AxCFN5Aaq8gRkeWKjFNtOzYFbqve/view?usp=drive_link](https://drive.google.com/file/d/1awD2AxCFN5Aaq8gRkeWKjFNtOzYFbqve/view?usp=sharing) 

Example on how to run Code from terminal to run the GUI:

```
python Run_GUI.py --config_file_path /Users/samgritz/Desktop/Ephys-Processing-GUI-main/Default_Metadata_new_stims_dev_080724.yaml --interactive --debug

```

## Example usage

Below contains analysis examples of a single cell. 

### Analyzing Action Potential Properties and Intrinsic Properties
GUI includes the ability to measure various properties of neuron action potentials (AP) and various intrinsic properties including: **AP threshold**, **AP firing rate**, **AP inter-spike-interval**, **AP halfwidth**, **AP size**, **After-Hyperpolarization size**, **Rheobase**, **input resistance**, **voltage sag**

**Definitions:**
- **AP Threshold:** The voltage (mV) at which the action potential is triggered. Measured as the max value of the first derivative before the peak of the action potential.
- **AP Firing Rate:** The firing rate or frequency is the number of action potentials per second. This is one of the key electrophysiological features of neurons.
- **AP inter-spike-interval (ISI):** A measure of the time between subsequent action potentials. This is typically used to measure _spike frequency adaptation_, which is a ratio between early ISI (e.g. the time between the first two action potentials) and late ISI (e.g. the last two action potentials) in the same trace.
- **AP halfwidth:** A measure of the full width (in ms) of the action potential at half height, which mainly depends density and type of potassium channels as well as the inactivation of sodium channels.
- **AP Size:** The amplitude or spike height measure from the action potential threshold.
- **After-Hyperpolarization Size:** After Hyperpolarization (AHP), or the trough of the voltage response, is measured as the voltage difference between the action potential threshold and the most negative voltage of the AHP. AHP regulated the intrinsic excitability of neurons, as increased AHP can slow the firing rate by increasing the _spike frequency adaptation_.
-  **Rheobase:** The minimum current required to elicit an action potential, which is an indication of somatic excitibility.
-  **Input Resistance:**: The **input resistance (Rin)** is the total resistance measured by the amplifier, which is the sum of the electrode resistance and, mainly, the **membrane resistance (Rm)**. Measured from a current injection of 50pA that is 100ms long (test pulse). Calculated as the change in voltage / change in current (R = V/I), where the change in voltage is end voltage - baseline voltage.
The input resistance is determined by the size of the cell and the number of open ion channels.
- **Voltage Sag** The initial negative voltage peak in response to a hyperpolarizing step is associated with the **hyperpolarization-activated current (Ih)**, an inward current mediated by HCN channels that start to open at approx. -60 mV. Calculated as (steady state voltage - trough of sag) / (baseline voltage - trough of sag).

### User Interface

![Using the GUI for processing](https://github.com/samuelgritz/Ephys-Processing-GUI/blob/32c9e5f577f55790a3f1e629e932e2e75185d693/User_Interface_GUI.png)

The user interface includes different frames for preprocessing and analyzing data as well as filling in metadata information if it does not already exist in the dataframe. 

### Plotting input resistance over time as measure of cell health:

![Plotting Input resistance across time in a single cell](https://github.com/samuelgritz/Ephys-Processing-GUI/blob/main/07022024_c2_BTSP_inp_res.png))

### Analyzing Voltage Trace at Rheobase

![Rheobase AP properties](https://github.com/samuelgritz/Ephys-Processing-GUI/blob/aee9aadf616171c8a3a273c90028d0259f1a1ddf/Analyze_Rheobase.png)

### Analyzing Postsynaptic Potentials and Complex Dendritic Spikes
GUI includes the ability to preprocess and clean voltage responses of **postsynaptic potentials** in several ways:

**Preprocess:**
Remove stimulus artifacts: With known stimulus times provided by the user via YAML file or through an automated function, the user can select the start and end duration of the interpolation that removes the artifact from the trace. 
- Remove noise: Gives the user the ability to remove noise manually from specific traces. The user chooses the time and duration of the artifact to be removed. Future iterations will include the ability to filter and preprocess using Lowpass, Highpass, 
Bandpass, and Stopband or notch filters.
- Interpolate Spikes: This function will clip action potentials if the goal is to analyze the underlying area of postsynaptic potentials without interference from somatic AP responses. The user defines the Action Potential spike width and duration of Action Potential, including afterhyperpolarization, to be interpolated over. 
  
**Postprocess:**
  - Partition trace: If the voltage traces include recordings with several different channels (i.e different input pathways), then traces will be partitioned to only analyze within a window surrounding the postsynaptic event.
  - Offset trace: Baseline offsets the partitioned trace to analyze the voltage response properties
  - Analyze EPSP peaks: Analyzes the amplitude of **Excitatory Postsynaptic Potentials (EPSPs)**.
  - Analyze Plateau: A more specific application to analyze complicated voltage responses known as complex **dendritic spikes** or **Ca 2+ spikes**

**Definitions:**
- **Excitatory Postsynaptic Potentials (EPSPs):** Electrical potential differences between the intracellular and the extracellular space. Excitatory means that the membrane voltage gets depolarized or most positive due to current flowing through receptors on the **postsynaptic membrane**. These postsynaptic electrical responses give a readout of synaptic transmission and can increase or decrease over time as a result of synaptic plasticity mechanisms such as short-term plasticity, long-term potentiation, and long-term depression.

- **Dendritic Spikes**: Dendritic spikes result from a supralinear increase in the membrane potential in the dendrites of neurons. These signals can be generated through sodium and calcium voltage-gated channels as well N-methyl-D-aspartate (NMDA) receptors. They play an important role in synaptic integration and neuronal computation as well synaptic plasticity.

### Analyzing and processing EPSPs

Users can choose to plot various stages of preprocessing and postprocessing steps. 

![EPSP analysis](https://github.com/samuelgritz/Ephys-Processing-GUI/blob/0e4fcd95b09e26f710b0161264aea20832be6265/Analyze_EPSP_trace.png)

### Analyzing and processing Complex Dendritic Spikes 

  















