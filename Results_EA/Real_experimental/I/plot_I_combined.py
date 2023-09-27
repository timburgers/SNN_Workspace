import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 30})
# all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]
all_files =["PID_with_I","PID_without_I","SNN"]
colors_files = ["tab:blue","tab:gray","tab:orange"]
labels = ["PID","PD","SNN PD + SNN I","Target"]
color_ref = "tab:red"
opacity = 0.7
stop_seconds= 350

# List of CSV files to open
fig,ax = plt.subplots(figsize=(20, 12))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# plt.gca().set_axis_off()
plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.08, 
            hspace = 0.2, wspace = 0.2)

ind =0
for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/I/"+file_name+"/"

    all_files = os.listdir(folder_csv)
    all_files = sorted(all_files)
    for file in all_files:
        df = pd.read_csv(folder_csv +file)
        time = df["time"].to_numpy()
        meas = df["meas"].to_numpy()
        plt.plot(time,meas, color = colors_files[ind], alpha= opacity, linewidth=2)

    ind +=1

ref = df["ref"].to_numpy()

plt.plot(time,ref, color = color_ref, linestyle = "--",linewidth = 3)
if stop_seconds != None:
    lim_t = stop_seconds
else:
    lim_t = 700
plt.xlim([0,lim_t])
plt.ylabel("Height [m]")
plt.xlabel("Time [s]")

lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors_files]
lines.append(Line2D([0], [0], color=color_ref, linewidth=3, linestyle="--"))

plt.legend(lines,labels,loc='upper left', frameon=True)



plt.savefig("/home/tim/Documents/plots_papers/Blimp_I_combined.pdf")
plt.show()