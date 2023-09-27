import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})

folder_csv = "Results_EA/Simulation/Recurrent_Adaptation/10s_steps/"
freq = 100
start_second = 0
end_second = 80

plot_ref = True
plot_meas = True
plot_u = False


# List of CSV files to open
all_files = os.listdir(folder_csv)
all_files = sorted(all_files)
all_files = ["R-IWTA-LIF.csv","R-LIF.csv","IWTA-LIF.csv","LIF.csv"]
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


# meas = np.array([[time]])
# ref = np.array([[time]])
# target = np.array([[time]])
# u = np.array([[time]])

# Loop through each CSV file and read into a DataFrame
plt.figure()
if plot_u:
    # plt.title("Zero input to SNN (100Hz)")
    plt.ylabel('Motor commmand [-]')
else:
    # plt.title("SNNs Controlling Double Integrator With Bias")
    plt.ylabel('Height [m]')
plt.xlabel('Time [s]')

plt.grid()
if plot_ref:
    plt.plot(time[start_second*freq:end_second*freq],df["ref"][start_second*freq:end_second*freq],color ='r',linestyle = "--", label = "Referece")

if plot_meas:
    for ind,csv_file in enumerate(all_files):
        df = pd.read_csv(folder_csv + csv_file)
        plt.plot(time[start_second*freq:end_second*freq],df["meas"][start_second*freq:end_second*freq], color = colors[ind], label = csv_file)

if plot_u:
    for ind,csv_file in enumerate(all_files):
        df = pd.read_csv(folder_csv + csv_file)
        plt.plot(time[start_second*freq:end_second*freq],df["u"][start_second*freq:end_second*freq], color = colors[ind], label = csv_file)
    

plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()