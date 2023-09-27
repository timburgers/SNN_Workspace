

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import gridspec
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 37})
# all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]
all_files =["PID_with_I_D","SNN_D"]
colors_files = ["tab:blue","tab:orange"]
labels = ["PID","SNN","Target"]
color_ref = "tab:red"
opacity = 0.4
start = 280
stop_seconds= 290
include_average_line = True
#A: 70-80
#B: 140-150
# List of CSV files to open
fig,ax = plt.subplots(figsize=(20, 12))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.gca().set_axis_off()
plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.09, 
            hspace = 0.2, wspace = 0.2)
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1,1,1]) 
ind = 0
ax0 = plt.subplot(gs[0])
ax1,ax2,ax3 = plt.subplot(gs[1],sharex = ax0), plt.subplot(gs[2],sharex = ax0), plt.subplot(gs[3],sharex = ax0)
ax_list = [ax0,ax1,ax2,ax3] 

ind =0
for file_name in all_files:
    ax0
    folder_csv = "Results_EA/Real_experimental/I/"+file_name+"/"
    time_values = [i/10 for i in range(0, stop_seconds*10+1)]
    df_pd_average_line = pd.DataFrame({'time': time_values})
    df_i_average_line = pd.DataFrame({'time': time_values})
    df_meas_average_line = pd.DataFrame({'time': time_values})
    all_files = os.listdir(folder_csv)
    all_files = sorted(all_files)
    for file in all_files:
        df = pd.read_csv(folder_csv +file)
        time = df["time"].to_numpy()
        meas = df["meas"].to_numpy()
        df_meas_current = df[['time', 'meas']]
        df_meas_average_line = pd.merge_asof(df_meas_average_line, df_meas_current, on="time")

        if file_name == "PID_with_I_D":
            u_pd = df["pid_pd"].to_numpy()*0.33
            u_i = df["pid_i"].to_numpy()*0.33
            u_pd = np.clip(u_pd, -3.3,3.3)
            df_pd_current = df[['time', 'pid_pd']]
            df_i_current = df[['time', 'pid_i']]
            df_pd_average_line = pd.merge_asof(df_pd_average_line, df_pd_current, on="time")
            df_i_average_line = pd.merge_asof(df_i_average_line, df_i_current, on="time")

        else:
            u_pd = df["snn_pd"].to_numpy()*0.33
            u_i = df["snn_i"].to_numpy()*0.33
            df_pd_current = df[['time', 'snn_pd']]
            df_i_current = df[['time', 'snn_i']]
            df_pd_average_line = pd.merge_asof(df_pd_average_line, df_pd_current, on="time")
            df_i_average_line = pd.merge_asof(df_i_average_line, df_i_current, on="time")


        ax0.plot(time,meas, color = colors_files[ind], alpha= opacity, linewidth=2)

        ax1.plot(time, u_pd,color = colors_files[ind], alpha= opacity)
        ax2.plot(time,u_i,color = colors_files[ind], alpha= opacity)
        ax3.plot(time, u_i+u_pd,color = colors_files[ind], alpha= opacity)
        ax3.hlines(y=0, xmin=0, xmax=stop_seconds, color="k")

    if include_average_line:
        avg_pd = df_pd_average_line.drop("time", axis=1).mean(axis=1).to_numpy()
        avg_i = df_i_average_line.drop("time", axis=1).mean(axis=1).to_numpy()
        avg_meas = df_meas_average_line.drop("time", axis=1).mean(axis=1).to_numpy()
        # avg_i = df_average_line.drop("time", axis=1).mean(axis=1).to_numpy()
        time = np.arange(0, stop_seconds+0.1, 0.1)
        avg_pd = np.clip(avg_pd, -10,10)
        ax0.plot(time,avg_meas, color = colors_files[ind],linewidth =6,alpha=1,zorder=100)
        ax1.plot(time,avg_pd*0.33, color = colors_files[ind],linewidth =6,alpha=1,zorder=100)
        ax2.plot(time,avg_i*0.33, color = colors_files[ind],linewidth =6,alpha=1,zorder=100)
        ax3.plot(time,(avg_pd*0.33)+(avg_i*0.33), color = colors_files[ind],linewidth =6,alpha=1,zorder=100)

    ind +=1

for ax in ax_list:
    if ax != ax3:
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0.2)
    
    ax.grid()


time = df["time"].to_numpy()
ref = df["ref"].to_numpy()

ax0.plot(time,ref, color = color_ref, linestyle = "--",linewidth = 3)
if stop_seconds != None:
    lim_t = stop_seconds
else:
    lim_t = 700
ax0.set_xlim([start,lim_t])
ax0.set_ylabel("Height [m]")
ax1.set_ylabel("$u_{pd}$ [-]")
ax2.set_ylabel("$u_{i}$ [-]")
ax3.set_ylabel("$u_{total}$ [-]")
ax.set_xlabel("Time [s]")

lines = [Line2D([0], [0], color=c, linewidth=6) for c in colors_files]
lines.append(Line2D([0], [0], color=color_ref, linewidth=6, linestyle="--"))

ax0.set_ylim([0,1.8])
ax1.set_ylim([-3.3,2])
ax2.set_ylim([0.2,1.5])
ax3.set_ylim([-3.3,2])

ax0.legend(lines,labels,loc='upper right', frameon=True, shadow=True,ncols=3)






# plt.savefig("/home/tim/Documents/plots_papers/controller_D.pdf")
plt.show()