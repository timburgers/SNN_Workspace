import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

all_files = ["PID_with_I","PID_without_I","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF","SNN"]
time_start = 60
time_end = 70

for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/I/"+file_name+"/"
    sse = []


    all_files = os.listdir(folder_csv)
    all_files = sorted(all_files)
    for file in all_files:
        df = pd.read_csv(folder_csv+file)
        # time = df["time"].to_numpy()
        start_index = (df.iloc[(df['time']-time_start).abs().argsort()[:1]]).index
        end_index = (df.iloc[(df['time']-time_end).abs().argsort()[:1]]).index
        error_range = df["error"].iloc[start_index[0]:end_index[0]]
        mean = error_range.abs().mean()
        sse.append(mean)

    print(file_name)
    print(sum(sse)/len(sse))




# plt.show()