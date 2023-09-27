import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 30})
all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]

for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/PD/"+file_name+"/"
    colors = ["tab:blue","tab:red"]

    # List of CSV files to open
    subfolders = [ f.name for f in os.scandir(folder_csv) if f.is_dir() ]

    fig,ax = plt.subplots(figsize=(20, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.08, 
                hspace = 0.2, wspace = 0.2)

    for folder in subfolders:
        all_files = os.listdir(folder_csv+folder)
        all_files = sorted(all_files)
        for file in all_files:
            df = pd.read_csv(folder_csv+folder + "/"+file)
            time = df["time"].to_numpy()
            meas = df["meas"].to_numpy()
            plt.plot(time,meas, color = colors[0])
        time = np.array([0,19.99,20.0,70.0])
        ref = np.array([0,0,df["ref"].iloc[50*10],df["ref"].iloc[50*10]])
        plt.plot(time,ref, color = colors[1], linestyle = "--")
    plt.xlim([0,70])
    plt.ylabel("Height [Δm]")
    plt.xlabel("Time [s]")

    plt.savefig("/home/tim/Documents/plots_papers/Blimp_PD_"+file_name +".pdf")

# plt.show()