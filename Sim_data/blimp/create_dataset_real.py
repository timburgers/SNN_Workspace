import os
import pandas as pd




file_name = "dataset_"
freq = 5
time_sim = 40
start_before_step=3 #s


original_folder = "/home/tim/SNN_Workspace/Sim_data/blimp/new/"
original_files = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f))]
original_files = sorted(original_files)
print(original_files)

# create dataset folder
if os.path.isdir(original_folder + "datasets/"): pass
else: os.mkdir(original_folder + "datasets/")


ind = 55
for file in original_files:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(original_folder + file)

    mask = df['h_ref'] != df['h_ref'].shift()
    rows_with_change = df.index[mask]
    rows_with_change = rows_with_change - rows_with_change[0]
    total_rows = df.shape[0]
    print("Rows with change: ", rows_with_change, " & Total rows = ", total_rows)

    for step_ind in rows_with_change:
        if step_ind-start_before_step*freq >= 0 and step_ind+(time_sim-start_before_step)*freq < total_rows:
            df_individual_step = df[step_ind-start_before_step*freq:step_ind+(time_sim-start_before_step)*freq]

            # Remove the initial offset of the integral of the integral output
            df_individual_step['pid_i'] = df_individual_step["pid_i"] - df_individual_step["pid_i"].iloc[0]

            # Set the first timestep of the d to zero
            df_individual_step['pid_d'].iloc[0] = 0
            df_individual_step['pid_pd'].iloc[0] = df_individual_step['error'].iloc[0]*10

            df_individual_step.to_csv(path_or_buf= original_folder + "datasets/" +file_name +str(ind) + ".csv", index=False)
            ind +=1







