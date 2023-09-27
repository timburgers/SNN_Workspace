import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec


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
plt.rcParams.update({'font.size': 35})
xlim = 420

file = "655-brisk-sun.csv"
folder_csv = "Results_EA/Simulation/PD/"
colors = ["tab:blue","tab:orange"]

df = pd.read_csv(folder_csv+file)
time = df["time"].to_numpy()
error = df["error"].to_numpy()
u = df["u"].to_numpy()*0.33
target = df["target_h"].to_numpy()*0.33


fig = plt.figure(figsize=(20, 12))
plt.subplots_adjust(top = 0.98, bottom = 0.13, right = 0.98, left = 0.08, 
            hspace = 0.2, wspace = 0.2)

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 

# First plot
ax0 = plt.subplot(gs[0])
ax0.hlines(y=0, xmin=0, xmax=xlim,colors="k",alpha = 0.8,linewidth=3)
ax0.plot(time,target, label = "Target PD",color = colors[0],linewidth=3)
ax0.plot(time,u,label = "SNN", color = colors[1],linewidth=3)

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
ax0.set_xlim([9,xlim])
ax0.set_ylim([-3.3,3.3])


ax1.set_ylabel("Error [m]")
ax1.set_ylim([-1.1,1.1])



# plt.savefig("/home/tim/Documents/plots_papers/Sim_PD.png")


plt.show()