import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import gridspec
from matplotlib.lines import Line2D

# fontsize= 40
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 37})
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
x_start =93
xlim = 102-x_start

# file = "151-sparkling-deluge.csv"
files = ["655-brisk-sun.csv","1039-celestial-wind.csv","967-confused-violet.csv","900-wobbly-totem.csv"]
names = ["LIF","IWTA-LIF","R-LIF","R-IWTA-LIF","Target PD","P reference"]
folder_csv = "Results_EA/Simulation/PD/"
colors = ["tab:green","tab:blue","tab:orange","tab:grey"]
color_ref = "tab:red"

fig = plt.figure(figsize=(20, 11))
plt.subplots_adjust(top = 0.98, bottom = 0.10, right = 0.98, left = 0.10, 
            hspace = 0.2, wspace = 0.2)

for ind,file in enumerate(files):
    df = pd.read_csv(folder_csv+file)
    time = df["time"].to_numpy()-x_start
    error = df["ref"].to_numpy()
    u = df["u"].to_numpy()*0.33
    




    window_size = 4
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    ma_u = moving_average(u,window_size)


    # First plot
    
    plt.plot(time[x_start*10:],u[x_start*10:],label = names[ind], color = colors[ind],linewidth=1,alpha = 0.25)
    plt.plot(time[x_start*10+window_size-1:],ma_u[x_start*10:],color = colors[ind],linewidth=6,zorder=99)

target = df["Target"].to_numpy()*0.33
ma_u = moving_average(target,window_size)
Kp= df["error"].to_numpy()*3.3

plt.plot(time[x_start*10:],target[x_start*10:],label = names[ind], color =color_ref ,linewidth=1,alpha = 0.25)
plt.plot(time[x_start*10+window_size-1:],ma_u[x_start*10:],color = color_ref,linewidth=6,zorder=100)
plt.plot(time[x_start*10:],Kp[x_start*10:],color = color_ref,linewidth=6,linestyle="--",zorder=100)

plt.hlines(y=0, xmin=0, xmax=xlim,colors="k",alpha = 0.8,linewidth=2)

lines = [Line2D([0], [0], color=c, linewidth=5) for c in colors]
lines.append(Line2D([0], [0], color=color_ref, linewidth=7))
lines.append(Line2D([0], [0], color=color_ref, linewidth=7, linestyle="--"))

plt.legend(lines,names,loc='upper right', frameon=True, shadow=True)


plt.subplots_adjust(hspace=.0)

plt.xlabel("Time [s]")
plt.ylabel("Motor command [-]")
plt.xlim([0,xlim])
plt.ylim([-0.5,1.7])
plt.grid()


plt.savefig("/home/tim/Documents/plots_papers/Sim_PD.pdf")
# plt.show()