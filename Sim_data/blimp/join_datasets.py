# Based on an error row, output the ideal P,I and D responses
import os
import pandas as pd


file_name = "dataset_"
freq = 10
Kp = 9
Ki = 0.1
Kd = 14
filter_d_peaks = True

limit_u = True
limit_p = True
lim_p = 15
limit_d = True
lim_d = 15
limit_pd = True
lim_pd = 15


original_folder = "/home/tim/SNN_Workspace/Sim_data/blimp/neutral_seperate_os2/"
original_files = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f))]
original_files = sorted(original_files)
print(original_files)

# create dataset folder
if os.path.isdir(original_folder + "datasets_filtered_d/"): pass
else: os.mkdir(original_folder + "datasets_filtered_d/")

df_final = pd.DataFrame(columns = ['time','meas','ref','error','pid_p','pid_i','pid_d','pid_pd', 'u', 'snn_p','snn_i','snn_d','snn_pd','snn_pid'])
previous_d = 0
previous_error = 0
integral = 0
file_ind = 0
index_prev = 0
index_overall = -1
for file in original_files:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(original_folder + file) 

    for index, row in df.iterrows():

        #Skip the first ten entries of the second file
        if file_ind == 1 and index in [0,1,2,3,4,5,6,7,8,9,10]:
            continue
        else: index_overall +=1

        # Fill in error
        time = (index_overall)/freq
        error = row["error"]
        try: ref=row["h_ref"]
        except: ref=row["ref"]

        # Fill in D
        derivative = (error - previous_error)*freq
        if index in [0,1] and file_ind==0:
            pid_d = 0
        else:
            pid_d = Kd*(derivative + previous_d)/2
        previous_d = derivative
        previous_error = error
        if abs(pid_d)>10:
            pid_d = pid_d/10

        # Fill in P
        pid_p = Kp * error

        #Fill in I
        integral = integral + row['error']/freq
        pid_i = Ki*integral
        df_final.loc[index_overall] = [time,0,ref,error,pid_p,pid_i,pid_d,pid_p+pid_d,pid_p+pid_i+pid_d,0,0,0,0,0]
    


    file_ind +=1

#Map the dc motor between -10 and 10 and between [10-100] to [10-15]
if limit_u:
    condition = df_final["u"] > 10
    df_final["u"][condition] = (df_final["u"][condition]-10)*5/90+10

    condition = df_final["u"] < -10
    df_final["u"][condition] = (df_final["u"][condition]+10)*5/90-10


#Map the dc motor between -10 and 10 and between [10-100] to [10-15]
if limit_p:
    condition = df_final["pid_p"] > lim_p
    df_final["pid_p"][condition] = (df_final["pid_p"][condition]-lim_p)*5/90+lim_p

    condition = df_final["pid_p"] < -lim_p
    df_final["pid_p"][condition] = (df_final["pid_p"][condition]+lim_p)*5/90-lim_p


#Map the dc motor between -10 and 10 and between [10-100] to [10-15]
if limit_d:
    condition = df_final["pid_d"] > lim_d
    df_final["pid_d"][condition] = (df_final["pid_d"][condition]-lim_d)*5/90+lim_d

    condition = df_final["pid_d"] < -lim_d
    df_final["pid_d"][condition] = (df_final["pid_d"][condition]+lim_d)*5/90-lim_d



#Map the dc motor between -10 and 10 and between [10-100] to [10-15]
if limit_pd:
    condition = df_final["pid_pd"] > lim_pd
    df_final["pid_pd"][condition] =(df_final["pid_pd"][condition]-lim_pd)*5/90+lim_pd

    condition = df_final["pid_pd"] < -lim_pd
    df_final["pid_pd"][condition] = (df_final["pid_pd"][condition]+lim_pd)*5/90-lim_pd
    

df_final.to_csv(path_or_buf= original_folder +"datasets_filtered_d/test_dataset.csv", index=False)


        
        





