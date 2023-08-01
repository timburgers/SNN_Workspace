import os
import pandas as pd




file_name = "test_dataset_"
freq = 5
time_sim = 200



original_folder = "/home/tim/SNN_Workspace/Sim_data/blimp/down/"
original_files = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f))]
original_files = sorted(original_files)
print(original_files)

# create dataset folder
if os.path.isdir(original_folder + "datasets/"): pass
else: os.mkdir(original_folder + "datasets/")


ind = 0
for file in original_files:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(original_folder + file)

    mask = df['h_ref'] != df['h_ref'].shift()
    rows_with_change = df.index[mask]
    rows_with_change = rows_with_change - rows_with_change[0]
    total_rows = df.shape[0]
    print("Rows with change: ", rows_with_change, " & Total rows = ", total_rows)

    for step_ind in rows_with_change:
        if step_ind+time_sim*freq < total_rows:
            df_individual_step = df[step_ind:step_ind+time_sim*freq]
            df_individual_step.to_csv(path_or_buf= original_folder + "datasets/" +file_name +str(ind) + ".csv", index=False)
            ind +=1







