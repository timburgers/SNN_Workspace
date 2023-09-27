import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 37})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]
all_files =["PID_with_I","PID_without_I","SNN"]
colors_files = ["tab:blue","tab:grey","tab:orange"]
labels = ["PID","PD" ,"SNN Controller","Target"]
color_ref = "tab:red"
opacity = 0.4
stop_seconds= 350
include_average_line = True

# List of CSV files to open
fig,ax = plt.subplots(figsize=(20, 12))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.gca().set_axis_off()
plt.subplots_adjust(top = 0.98, bottom = 0.11, right = 0.98, left = 0.07, 
            hspace = 0.2, wspace = 0.2)

ind =0
for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/I/"+file_name+"/"
    time_values = [i/10 for i in range(0, stop_seconds*10+1)]
    df_average_line = pd.DataFrame({'time': time_values})
    all_files = os.listdir(folder_csv)
    all_files = sorted(all_files)
    for file in all_files:
        df = pd.read_csv(folder_csv +file)
        time = df["time"].to_numpy()
        meas = df["meas"].to_numpy()
        plt.plot(time,meas, color = colors_files[ind], alpha= opacity, linewidth=2)
        df_current = df[['time', 'meas']]
        df_average_line = pd.merge_asof(df_average_line, df_current, on="time")

    if include_average_line:
        row_avg = df_average_line.drop("time", axis=1).mean(axis=1).to_numpy()
        time = np.arange(0, stop_seconds+0.1, 0.1)
        ax.plot(time,row_avg, color = colors_files[ind],linewidth =6,alpha=1,zorder=100)

    ind +=1

ref = df["ref"].to_numpy()
time = df["time"].to_numpy()
plt.plot(time,ref, color = color_ref, linestyle = "--",linewidth = 6)
if stop_seconds != None:
    lim_t = stop_seconds
else:
    lim_t = 700
plt.ylim([0,1.75])
plt.xlim([0,lim_t])
plt.ylabel("Height [m]")
plt.xlabel("Time [s]")
plt.grid()

lines = [Line2D([0], [0], color=c, linewidth=8) for c in colors_files]
lines.append(Line2D([0], [0], color=color_ref, linewidth=8, linestyle="--"))

plt.legend(lines,labels,loc='upper left', frameon=True, shadow=True,ncol=2)



plt.savefig("/home/tim/Documents/plots_papers/Blimp_I_combined_overleaf.png")
# plt.show()