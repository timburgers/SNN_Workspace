import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D

# mpl.use('Agg')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 37})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]
all_files = ["PID","LIF"]
colors_files = ["tab:blue","tab:orange"]
labels = ["PD","LIF", "Target"]
color_ref = "tab:red"
all_in_one_plot = True
opacity = 0.5
include_average_line = True


if all_in_one_plot == True:
    fig,ax = plt.subplots(figsize=(20, 1))
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.09, 
            hspace = 0.2, wspace = 0.2)


ind =0
for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/PD/"+file_name+"/"
    

    # List of CSV files to open
    subfolders = [ f.name for f in os.scandir(folder_csv) if f.is_dir() ]

    if all_in_one_plot == False:
        fig,ax = plt.subplots(figsize=(20, 12))
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # plt.gca().set_axis_off()
        plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.08, 
                hspace = 0.2, wspace = 0.2)

    for folder in subfolders:
        all_files = os.listdir(folder_csv+folder)
        all_files = sorted(all_files)
        average_line = []
        for file in all_files:
            df = pd.read_csv(folder_csv+folder + "/"+file)
            time = df["time"].to_numpy()
            meas = df["meas"].to_numpy()
            plt.plot(time,meas, color = colors_files[ind],alpha=opacity)
            average_line.append(meas.tolist())
        if include_average_line:
            average_line_arr = np.asarray(average_line).mean(axis=0)
            ax.plot(time,average_line_arr, color = colors_files[ind],linewidth =8,alpha=1,zorder=99)


        time = np.array([0,19.99,20.0,70.0])
        ref = np.array([0,0,df["ref"].iloc[50*10],df["ref"].iloc[50*10]])
        plt.plot(time,ref, color = color_ref, linestyle = "--",linewidth=6)
    ind +=1

plt.xlim([0,70])
plt.ylabel("Height [Δm]")
plt.xlabel("Time [s]")
plt.grid()

lines = [Line2D([0], [0], color=c, linewidth=8) for c in colors_files]
lines.append(Line2D([0], [0], color=color_ref, linewidth=8, linestyle="--"))


plt.legend(lines,labels,loc='upper left', frameon=True,shadow=True)

plt.savefig("/home/tim/Documents/plots_papers/Blimp_PD_combined.pdf")

# plt.show()