import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D



# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams.update({'font.size': 40})
# xlim = 420

# file = "655-brisk-sun.csv"
# folder_csv = "Results_EA/Simulation/PD/"
# colors = ["tab:blue","tab:orange"]


# fig,ax = plt.subplots(figsize=(20, 12))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.hlines(y=0, xmin=0, xmax=xlim,colors="k")
# plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.08, 
#             hspace = 0.2, wspace = 0.2)


# df = pd.read_csv(folder_csv+file)
# time = df["time"].to_numpy()
# error = df["error"].to_numpy()
# u = df["u"].to_numpy()*0.33
# target = df["target_h"].to_numpy()*0.33

# plt.plot(time,target, label = "Target PD",color = colors[0])
# plt.plot(time,u,label = "SNN", color = colors[1])


# plt.xlabel("Time [s]")
# plt.ylabel("Motor command [V]")
# plt.xlim([0,xlim])
# plt.ylim([-3.3,3.3])
# plt.legend(loc='upper right', frameon=True)

# plt.savefig("/home/tim/Documents/plots_papers/Sim_PD.pdf")


# # plt.show()



plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 37})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
xlim = 500

# file = "119-snowy-sun.csv"
files = ["151-sparkling-deluge.csv","121-icy-gorge.csv","101-vibrant-paper.csv"]
names = ["IWTA-LIF","R-LIF","R-IWTA-LIF","Target I"]
folder_csv = "Results_EA/Simulation/I/"
colors = ["tab:blue","tab:orange","tab:grey"]
color_ref = "tab:red"

fig = plt.figure(figsize=(20, 11.5))
plt.subplots_adjust(top = 0.98, bottom = 0.10, right = 0.98, left = 0.07, 
            hspace = 0.2, wspace = 0.2)

for ind,file in enumerate(files):
    df = pd.read_csv(folder_csv+file)
    time = df["time"].to_numpy()
    error = df["ref"].to_numpy()
    u = df["u"].to_numpy()
    




    window_size = 20
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    ma_u = moving_average(u,window_size)


    # First plot
    
    plt.plot(time,u,label = names[ind], color = colors[ind],linewidth=2,alpha = 0.25)
    plt.plot(time[window_size-1:],ma_u,color = colors[ind],linewidth=5,zorder=99)

target = df["Target"].to_numpy()
plt.plot(time,target,linestyle="--",color = color_ref,linewidth=5,zorder=100)
plt.hlines(y=0, xmin=0, xmax=xlim,colors="k",alpha = 0.8,linewidth=2)

lines = [Line2D([0], [0], color=c, linewidth=7) for c in colors]
lines.append(Line2D([0], [0], color=color_ref, linewidth=7, linestyle="--"))

plt.legend(lines,names,loc='upper left', frameon=True, shadow=True)


plt.subplots_adjust(hspace=.0)

plt.xlabel("Time [s]")
plt.ylabel("Motor command [-]")
plt.xlim([9,xlim])
plt.ylim([-2.7,3.3])
plt.grid()


plt.savefig("/home/tim/Documents/plots_papers/Sim_I.pdf")
# plt.show()