import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

all_files = ["PID_with_I","PID_without_I","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF","SNN"]


for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/I/"+file_name+"/"
    mse = []


    all_files = os.listdir(folder_csv)
    all_files = sorted(all_files)
    for file in all_files:
        df = pd.read_csv(folder_csv+file)
        # time = df["time"].to_numpy()
        meas = df["meas"].to_numpy()
        ref = df["ref"].to_numpy()
        mse.append(np.sqrt((np.square(meas - ref)).mean()))

    print(file_name)
    print(sum(mse)/len(mse))




# plt.show()