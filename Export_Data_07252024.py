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

#Tell the user to enter the directory path for an individual cell with a given date that will contain multiple experiments
# myDir = '/Users/samgritz/Library/CloudStorage/GoogleDrive-gritz122@gmail.com/My Drive/Rutgers/Milstein_Lab_Data/GNB1_WT_Project/WT_littermate_Cells_42224_on'
config_file_path = '/Users/samgritz/Desktop/Rutgers/Milstein_Lab/Code/Rutgers-Neuroscience-PhD/Ephys_Analysis/Updated_Code/Intracellular_Analysis/dataframe_defaults.yaml'
#Now get the .pxp files from this directory and load those files

# save_dir = '/Users/samgritz/Library/CloudStorage/GoogleDrive-gritz122@gmail.com/My Drive/Rutgers/Milstein_Lab_Data/GNB1_WT_Project/New_Analysis_Processed_Data/Test'

myDir = simpledialog.askstring("File Path", "Enter a directory Path (Please make sure no log0.pxp and backup.pxp files exist and that files end in index not date):")
save_dir = simpledialog.askstring("File Path", "Enter a directory Path to save the data (Please make sure no log0.pxp and backup.pxp files exist and that files end in index not date):")