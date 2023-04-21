import matplotlib.pylab as plt
import pandas as pd
import numpy as np

data_dir = "Sim_data/derivative/dt0.002_norm_neg"
number_dataset = 3
input_column = 1
target_column = 1   #1=P, 2=D

print("number = ", number_dataset)
time_step = 0.002
total_path = data_dir + "/dataset_" + str(number_dataset)+ ".csv"

input_data = pd.read_csv(total_path, usecols=[input_column], header=None, skiprows=[0])
target_data = pd.read_csv(total_path, usecols=[target_column], header=None, skiprows=[0])
input_data = input_data.values[:,0] 
target_data = target_data.values[:,0]	

time = np.arange(0,len(input_data)*time_step,time_step)

plt.plot(time,input_data, label= "Input", color='k')
plt.plot(time, target_data, label="Target",color = 'r')
plt.legend()
plt.grid()
plt.show()