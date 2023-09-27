import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib.lines import Line2D


# Red       #ee4035
# Orange    #f37736
# Yellow    #fdf498
# Green     #7bc043
# Blue      #0392cf

include_PD_in_all_plots = True
include_average_line = True

# plt.style.use('ggplot')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 50})
# all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]
names =["LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]
text_names =["A) LIF","B) IWTA-LIF","C) R-LIF","D) R-IWTA-LIF"]


fig = plt.figure(figsize=(30, 40))
plt.subplots_adjust(top = 0.98, bottom = 0.05, right = 0.98, left = 0.08, 
            hspace = 0.2, wspace = 0.2)

gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1,1,1]) 
ind = 0
ax0 = plt.subplot(gs[0])
ax1,ax2,ax3 = plt.subplot(gs[1],sharex = ax0), plt.subplot(gs[2],sharex = ax0), plt.subplot(gs[3],sharex = ax0)
ax_list = [ax0,ax1,ax2,ax3] 

# plt.style.use('ggplot')
if include_average_line:
    opacity = [0.4,1.0]
else:
    opacity = [0.8]


for file_name in names:
    ax = ax_list[ind]
    ax.grid()

    #colours [SNN,ref, PD]
    # colors = ["#0392cf","#f37736"]
    colors = ["tab:orange","tab:blue"]
    # ref_color = "#ee4035"
    ref_color = "tab:red"

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    if ind!=3:
        # ax.spines["bottom"].set_linewidth(5)
        plt.setp(ax.get_xticklabels(), visible=False)
    plt.subplots_adjust(hspace=.0)
    # ax.text(2, 1.0, text_names[ind], fontsize = 70)
    ax.set_ylabel("Height [Δm]")

    if include_PD_in_all_plots:
        folder_csv = "Results_EA/Real_experimental/PD/PID/"
        subfolders = [ f.name for f in os.scandir(folder_csv) if f.is_dir() ]
        
        for folder in subfolders:
            all_files = os.listdir(folder_csv+folder)
            all_files = sorted(all_files)
            average_line = []
            for file in all_files:
                df = pd.read_csv(folder_csv +folder+"/"+file)
                time = df["time"].to_numpy()
                meas = df["meas"].to_numpy()
                ax.plot(time,meas, color = colors[1],linewidth =4, alpha=opacity[0])

                average_line.append(meas.tolist())
            if include_average_line:
                average_line_arr = np.asarray(average_line).mean(axis=0)
                ax.plot(time,average_line_arr, color = colors[1],linewidth =8,alpha=opacity[1],zorder=99)



    folder_csv = "Results_EA/Real_experimental/PD/"+file_name+"/"
    subfolders = [ f.name for f in os.scandir(folder_csv) if f.is_dir() ]
    for folder in subfolders:
        all_files = os.listdir(folder_csv+folder)
        all_files = sorted(all_files)
        average_line = []
        for file in all_files:
            df = pd.read_csv(folder_csv +folder+"/"+file)
            time = df["time"].to_numpy()
            meas = df["meas"].to_numpy()
            ax.plot(time,meas, color = colors[0],linewidth =4, alpha=opacity[0])

            average_line.append(meas.tolist())

        if include_average_line:
            average_line_arr = np.asarray(average_line).mean(axis=0)
            ax.plot(time,average_line_arr, color = colors[0],linewidth =8,alpha=opacity[1],zorder=100)


        # Add the reference lines
        time = np.array([0,19.99,20.0,70.0])
        ref = np.array([0,0,df["ref"].iloc[50*10],df["ref"].iloc[50*10]])
        ax.plot(time,ref, color = ref_color ,linewidth =5, linestyle = "--")
    
    lines = [Line2D([0], [0], color=c, linewidth=8) for c in colors]
    # lines.append(Line2D([0], [0], color=color_ref, linewidth=3, linestyle="--"))
    label = [file_name,"PD"]
    ax.legend(lines,label,loc='upper left', frameon=True, shadow=True)

    ind +=1


plt.xlim([0,70])
# plt.ylabel("Height [m]")
plt.xlabel("Time [s]")

plt.savefig("/home/tim/Documents/plots_papers/Blimp_D_combined_all.pdf")
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