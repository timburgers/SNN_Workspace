import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec



plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 35})
xlim = 100

file = "101-vibrant-paper.csv"
folder_csv = "Results_EA/Simulation/I/"
colors = ["tab:blue","tab:orange"]

df = pd.read_csv(folder_csv+file)
time = df["time"].to_numpy()
error = df["error"].to_numpy()
u = df["u"].to_numpy()


window_size = 15

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

fig = plt.figure(figsize=(20, 12))
plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.10, 
            hspace = 0.2, wspace = 0.2)

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 

# First plot
ax0 = plt.subplot(gs[0])

ma_u = moving_average(u,window_size)

ax0.hlines(y=0, xmin=0, xmax=xlim,colors="k",alpha = 0.8,linewidth=1)
ax0.plot(time[window_size-1:],ma_u, label = "Moving averge",color = colors[0],linewidth=2)
ax0.plot(time,u,label = "SNN", color = colors[1],linewidth=1,alpha = 0.8)




#Second error plot
ax1 = plt.subplot(gs[1], sharex = ax0)
ax1.plot(time,error, color = "tab:gray",linewidth=3)
ax1.hlines(y=0, xmin=0, xmax=xlim,colors="k",alpha = 0.8)




plt.setp(ax0.get_xticklabels(), visible=False)

# remove last tick label for the second subplot
yticks = ax1.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)

ax0.legend(loc='upper right', frameon=True)
plt.subplots_adjust(hspace=.0)

ax1.set_xlabel("Time [s]")
ax0.set_ylabel("Motor command [V]")
ax0.set_xlim([0,xlim])
ax0.set_ylim([0,3])


ax1.set_ylabel("Error [m]")
ax1.set_ylim([0,0.4])



# plt.savefig("/home/tim/Documents/plots_papers/Sim_I.pdf")


plt.show()