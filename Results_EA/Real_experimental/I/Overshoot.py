import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

all_files = ["PID_with_I","LIF","IWTA-LIF","R-LIF","R-IWTA-LIF","SNN"]
check_seconds = 10

for file_name in all_files:
    folder_csv = "Results_EA/Real_experimental/I/"+file_name+"/"
    os_list = []
    us_list = []


    all_files = os.listdir(folder_csv)
    all_files = sorted(all_files)
    for file in all_files:
        df = pd.read_csv(folder_csv+file)
        # time = df["time"].to_numpy()
        df = df.iloc[:350*10]

        mask = df['ref'] != df['ref'].shift()
        rows_with_change = df.index[mask]
        rows_with_change = rows_with_change - rows_with_change[0]
        

        for step in rows_with_change:
            df_small = df.iloc[step:step+30*10]
            delta_change = df["ref"].iloc[step]-df["ref"].iloc[step-1] 
            if np.sign(delta_change)==1:
                df_small =df_small["error"].clip(upper=0)
                overshoot = df_small.min()

                if overshoot == 0:
                    pass
                else:
                    os_list.append(overshoot)

            if np.sign(delta_change)==-1:
                df_small =df_small["error"].clip(lower=0)
                undershoot = df_small.max()

                if undershoot == 0:
                    pass
                else:
                    us_list.append(undershoot)


    print(file_name)
    print("Overshoot = ",sum(os_list)/len(os_list))
    print("Undershoot = ",sum(us_list)/len(us_list))




# plt.show()