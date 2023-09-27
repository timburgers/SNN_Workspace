import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

all_files = ["PID","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF"]


for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/PD/"+file_name+"/"
    mse = []

    # List of CSV files to open
    subfolders = [ f.name for f in os.scandir(folder_csv) if f.is_dir() ]

    for folder in subfolders:
        all_files = os.listdir(folder_csv+folder)
        all_files = sorted(all_files)
        for file in all_files:
            df = pd.read_csv(folder_csv+folder + "/"+file)
            # time = df["time"].to_numpy()
            meas = df["meas"].to_numpy()
            ref = df["ref"].to_numpy()
            mse.append(np.sqrt((np.square(meas - ref)).mean()))

    print(file_name)
    print(sum(mse)/len(mse))




# plt.show()