import os
import pandas as pd



multiply_freq = 2           # The number at which the frequency of the signal is increased         


original_folder = "/home/tim/SNN_Workspace/Sim_data/blimp/neutral_seperate_os/"
new_folder = "/home/tim/SNN_Workspace/Sim_data/blimp/neutral_seperate_os/new/"
original_files = [f for f in os.listdir(original_folder ) if os.path.isfile(os.path.join(original_folder, f))]
original_files = sorted(original_files)
print(original_files)

# create dataset folder
# if os.path.isdir(new_folder + "datasets/"): pass
# else: 
#     os.mkdir(new_folder)
#     os.mkdir(new_folder + "datasets/")




ind = 0
for file in original_files:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(original_folder + file)
    df_final = pd.DataFrame(columns = df.columns.values)

    # Get the steps and do not intrapolate when a step occurs
    mask = df['h_ref'] != df['h_ref'].shift()
    rows_with_change = df.index[mask]
    rows_with_change = rows_with_change - rows_with_change[0]
    total_rows = df.shape[0]
    print("Rows with change: ", rows_with_change, " & Total rows = ", total_rows)



    for index, row in df.iterrows():
        if index == 0: 
            df_final.loc[index] = row
            prev_row = row

        else:
            # Skip intrapolation when step occurs
            if index not in rows_with_change:
                df_final.loc[index*multiply_freq-1] = prev_row + (row - prev_row)/(multiply_freq)
            df_final.loc[index*multiply_freq]   = row

            prev_row =row
            
    df_final.to_csv(path_or_buf= new_folder +file, index=False)
    ind +=1







