import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D

# Red       #ee4035
# Orange    #f37736
# Yellow    #fdf498
# Green     #7bc043
# Blue      #0392cf

include_PID_in_all_plots = True
include_average_line = True

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 50})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]
names =["IWTA-LIF","R-LIF","R-IWTA-LIF"]
text_names =["A) IWTA-LIF","C) R-LIF","D) R-IWTA-LIF"]

if include_average_line:
    opacity = [0.4,1.0]
else:
    opacity = [0.8]


fig = plt.figure(figsize=(25, 30))
plt.subplots_adjust(top = 0.98, bottom = 0.06, right = 0.98, left = 0.06, 
            hspace = 0.2, wspace = 0.2)

gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1,1]) 
ind = 0
ax0 = plt.subplot(gs[0])
ax1,ax2 = plt.subplot(gs[1],sharex = ax0), plt.subplot(gs[2],sharex = ax0)
ax_list = [ax0,ax1,ax2] 



for file_name in names:
    ax = ax_list[ind]
    ax.grid()
    colors = ["tab:orange","tab:blue"]
    ref_color = "tab:red"

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    if ind!=2:
        # ax.spines["bottom"].set_linewidth(5)
        plt.setp(ax.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=.0)
    # ax.text(10, 1.35, text_names[ind], fontsize = 70)
    ax.set_ylabel("Height [m]")
    

    #Plot PID
    if include_PID_in_all_plots:
        folder_csv = "Results_EA/Real_experimental/I/PID_with_I/"
        all_files = os.listdir(folder_csv)
        all_files = sorted(all_files)
        time_values = [i/10 for i in range(0, 3501)]
        df_average_line = pd.DataFrame({'time': time_values})
        for file in all_files:
            df = pd.read_csv(folder_csv +file)
            time = df["time"].to_numpy()
            meas = df["meas"].to_numpy()
            ax.plot(time,meas, color = colors[1],linewidth =4,alpha=opacity[0])
            df_current = df[['time', 'meas']]
            df_average_line = pd.merge_asof(df_average_line, df_current, on="time")

        
        if include_average_line:
            row_avg = df_average_line.drop("time", axis=1).mean(axis=1).to_numpy()
            time = np.arange(0, 350.1, 0.1)
            ax.plot(time,row_avg, color = colors[1],linewidth =8,alpha=opacity[1],zorder=99)


    #Plot SNN
    folder_csv = "Results_EA/Real_experimental/I/"+file_name+"/"
    all_files = os.listdir(folder_csv)
    all_files = sorted(all_files)
    time_values = [i/10 for i in range(0, 3501)]
    df_average_line = pd.DataFrame({'time': time_values})
    for file in all_files:
        df = pd.read_csv(folder_csv +file)
        time = df["time"].to_numpy()
        meas = df["meas"].to_numpy()
        ax.plot(time,meas, color = colors[0],linewidth =4, alpha=opacity[0])
        
        #For the avergae line'
        df_current = df[['time', 'meas']]
        df_average_line = pd.merge_asof(df_average_line, df_current, on="time")

    if include_average_line:
        row_avg = df_average_line.drop("time", axis=1).mean(axis=1).to_numpy()
        time = np.arange(0, 350.1, 0.1)
        ax.plot(time,row_avg, color = colors[0],linewidth =8,alpha=opacity[1],zorder=100)

    time = df["time"].to_numpy()
    ref = df["ref"].to_numpy()
    ax.plot(time,ref, color = ref_color,linewidth =6, linestyle = "--")

    lines = [Line2D([0], [0], color=c, linewidth=8) for c in colors]
    # lines.append(Line2D([0], [0], color=color_ref, linewidth=3, linestyle="--"))
    label = ["PD + "+file_name,"PID"]
    ax.legend(lines,label,loc='upper left', frameon=True, shadow=True)
    

    ind +=1


plt.xlim([0,350])
# plt.ylabel("Height [m]")
plt.xlabel("Time [s]")
# plt.grid()
plt.savefig("/home/tim/Documents/plots_papers/Blimp_I_combined_all.pdf")
# plt.show()





# # First plot

# ax0.hlines(y=0, xmin=0, xmax=xlim,colors="k",alpha = 0.8,linewidth=1)
# ax0.plot(time,target, label = "Target I",color = colors[0],linewidth=2)
# ax0.plot(time,u,label = "SNN", color = colors[1],linewidth=1,alpha = 0.8)

# #Second error plot
# ax1 = plt.subplot(gs[1], sharex = ax0)
# ax1.plot(time,error, color = "tab:gray",linewidth=3)
# ax1.hlines(y=0, xmin=0, xmax=xlim,colors="k",alpha = 0.8)

# plt.setp(ax0.get_xticklabels(), visible=False)

# # remove last tick label for the second subplot
# yticks = ax1.yaxis.get_major_ticks()
# yticks[-1].label1.set_visible(False)

# ax0.legend(loc='upper right', frameon=True)
# plt.subplots_adjust(hspace=.0)

# ax1.set_xlabel("Time [s]")
# ax0.set_ylabel("Motor command [V]")
# ax0.set_xlim([9,xlim])
# ax0.set_ylim([-3.3,3.3])


# ax1.set_ylabel("Error [m]")
# ax1.set_ylim([-0.25,0.25])



# plt.savefig("/home/tim/Documents/plots_papers/Sim_I.pdf")


# # plt.show()