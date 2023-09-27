import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 30})
# all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]
all_files =["IWTA-LIF","R-IWTA-LIF","LIF","R-LIF"]

for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/I/"+file_name+"/"
    colors = ["tab:blue","tab:red"]

    # List of CSV files to open
    fig,ax = plt.subplots(figsize=(20, 8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.08, 
                hspace = 0.2, wspace = 0.2)

    all_files = os.listdir(folder_csv)
    all_files = sorted(all_files)
    for file in all_files:
        df = pd.read_csv(folder_csv +file)
        time = df["time"].to_numpy()
        meas = df["meas"].to_numpy()
        plt.plot(time,meas, color = colors[0])
    ref = df["ref"].to_numpy()

    plt.plot(time,ref, color = colors[1], linestyle = "--")
    plt.xlim([0,350])
    plt.ylabel("Height [m]")
    plt.xlabel("Time [s]")

    plt.savefig("/home/tim/Documents/plots_papers/Blimp_I_"+file_name +".pdf")
    plt.show()