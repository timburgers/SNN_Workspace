import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 16})

folder_csv = "Results_EA/Real_experimental/PD/"
freq = 10
before_step = 20
after_step = 50
steps = [1.0,0.5,0,-0.5,-1.0]
folder_name = "20s_step_50s"


# List of CSV files to open
all_csv = [f.split(".")[0] for f in os.listdir(folder_csv) if os.path.isfile(os.path.join(folder_csv, f))]
all_csv = sorted(all_csv)


folders=[]
for step in steps:
    if os.path.isdir(folder_csv + folder_name+"_"+str(step)+"m"): pass
    else: os.mkdir(folder_csv + folder_name+"_"+str(step)+"m")
    folders.append(folder_csv + folder_name+"_"+str(step)+"m")


# Create an empty list to store DataFrames
dataframes = []
for file in all_csv:
    #Get the maximum number of rows
    df = pd.read_csv(folder_csv + file + ".csv")

    # Find the rows where the reference input changes
    mask = df['ref'] != df['ref'].shift()
    rows_with_change = df.index[mask]
    rows_with_change = rows_with_change - rows_with_change[0]
    total_rows = df.shape[0]

    for step_ind in rows_with_change:
        if step_ind-before_step*freq >= 0 and step_ind+after_step*freq < total_rows:
            step_value = df['ref'].iloc[step_ind] - df['ref'].iloc[step_ind-1]
            df_individual_step = df[step_ind-before_step*freq:step_ind+after_step*freq]                
            index_steps = steps.index(step_value)

            df_individual_step["time"] = df_individual_step["time"] - df_individual_step["time"].iloc[0]
            df_individual_step["meas"] = df_individual_step["meas"] - df['ref'].iloc[step_ind-1]
            df_individual_step["ref"] = df_individual_step["ref"] - df['ref'].iloc[step_ind-1]
            
            df_individual_step.to_csv(path_or_buf= folders[index_steps]+"/"+file+"_"+str(step_ind) + ".csv", index=False)
    
    if total_rows-(before_step+after_step)*freq>= step_ind:
        df_individual_step = df[total_rows-(before_step+after_step)*freq:total_rows]              
        index_steps = steps.index(0)
        df_individual_step["time"] = df_individual_step["time"] - df_individual_step["time"].iloc[0]
        df_individual_step["meas"] = df_individual_step["meas"] - df['ref'].iloc[step_ind]
        df_individual_step["ref"] = df_individual_step["ref"] - df['ref'].iloc[step_ind]

        df_individual_step.to_csv(path_or_buf= folders[index_steps]+"/"+file+"_"+str(step_ind) + ".csv", index=False)
    





    





