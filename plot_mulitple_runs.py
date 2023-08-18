import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})

folder_csv = "Results_EA/Simulation/Recurrent_Adaptation/5s_steps/"
freq = 100
start_second = None
end_second = 80


# List of CSV files to open
all_files = os.listdir(folder_csv)
all_files = sorted(all_files)
all_files = ["R-ALIF.csv","R-LIF.csv","ALIF.csv","LIF.csv"]
colors = ["tab:blue","tab:orange", "tab:green", "tab:gray"]

num_files = len(all_files)

# Create an empty list to store DataFrames
dataframes = []

#Get the maximum number of rows
df = pd.read_csv(folder_csv + all_files[0])
time = df["time"]

#Set the start and end point is None is used
if start_second ==None: start_second = 0
if end_second == None: end_second =int(len(time)/freq)


meas = np.array([[time]])
ref = np.array([[time]])
target = np.array([[time]])
u = np.array([[time]])

# Loop through each CSV file and read into a DataFrame
plt.figure()

plt.title("SNNs Controlling Double Integrator With Bias")
plt.xlabel('Time [s]')
plt.ylabel('Height [m]')
plt.grid()
plt.plot(time[start_second*freq:end_second*freq],df["ref"][start_second*freq:end_second*freq],color ='r',linestyle = "--", label = "Referece")
for ind,csv_file in enumerate(all_files):
    df = pd.read_csv(folder_csv + csv_file)
    plt.plot(time[start_second*freq:end_second*freq],df["meas"][start_second*freq:end_second*freq], color = colors[ind], label = csv_file)
    

plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()