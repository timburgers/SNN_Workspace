# Based on an error row, output the ideal P,I and D responses
import os
import pandas as pd

new_folder = "datasets_new/"

original_folder = "/home/tim/SNN_Workspace/Sim_data/blimp/neutral_seperate_os2/"
original_files = [f for f in os.listdir(original_folder+"datasets") if os.path.isfile(os.path.join(original_folder+"datasets", f))]
original_files = sorted(original_files)
print(original_files)

# create dataset folder
if os.path.isdir(original_folder + new_folder): pass
else: os.mkdir(original_folder + new_folder)


for file in original_files:
    # Read the CSV file into a pandas DataFrame
    df_original = pd.read_csv(original_folder+"datasets/" + file) 
    df_final = df_original
    df_final["pid_i"] = df_original["pid_i"]*7.5

    df_final.to_csv(path_or_buf= original_folder +new_folder+file, index=False)


        
        





