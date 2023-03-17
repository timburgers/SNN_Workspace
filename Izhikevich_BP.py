# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
import torch.nn as nn
from SNN_Izhickevich import Izhikevich_SNN
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
import matplotlib.pyplot as plt
import numpy as np
from Coding.Encoding import input_cur
from Coding.Decoding import sliding_window, Linear_LI_filter
from Backprop_SNN import train_SNN
import pandas as pd

# # Proportional
# neuron_par = dict(a = 0.1, b = 0.222, c = -61.6, d = 0, threshold = 30)

# Derivative
neuron_par = dict(a = 0.0105, b = 0.656, c = -55.0, d = 1.92, threshold = 30)
# neuron_par = dict(a = 0.0000105, b = 0.000656, c = -0.055, d = 0.00192, threshold = 0.030)

# # Integral
# neuron_par = dict(a = 0.0158, b = 0.139, c = -70.0, d = 1.06, threshold = 30)


neurons = 1
weight_syn = 1.
time_step = 0.002

# read input signal from data file
input = pd.read_csv("Sim_data/derivative/dataset_derivative_sin_0.csv", usecols=[1],header=None, skiprows=[0])
input = torch.tensor(input.values).float() 																			# convert from pandas df to torch tensor and floats 
input = torch.tile(input,(1,neurons))																				# convert from (input_signals,1) to (input_signals,neurons)
print(input)

# Retrieve timestep and time files from data file
# time_step = pd.read_csv("Sim_data/derivative/dataset_derivative_sin_0.csv", usecols=[0],skiprows=[0],nrows=1 ,header=None)
# time_step = time_step.values[0,0]
# time = (pd.read_csv("Sim_data/derivative/dataset_derivative_sin_0.csv", usecols=[0],skiprows=[0] ,header=None)).to_numpy()

#Initialize SNN
SNN_izhik = Izhikevich_SNN(neurons,neuron_par,weight_syn,time_step)

# trained_network = train_SNN(SNN_izhik,"Sim_data/derivative",3,500,0.8)
# print(trained_network.params_learnable)



# Initialize the states(u, v, s)
states = torch.zeros(3,neurons)
states[1]= -70			# To prevent first spike to happen (set v to -82.65)
states[0]= -10

# Call forward function
spike_snn, state_snn, decoded = SNN_izhik(input,states)


#Set target for the output decoded data, since input.detach.numpy is of shape (x,1) --> axis of diff is 0
output_target = np.diff(input[:,0].detach().numpy(), axis=0)/time_step


################################# Creating Plots ################################################

# Convert all torch.tensors to numpu element
recovery = state_snn[:,0,0].detach().numpy()
mem_pot = state_snn[:,1,0].detach().numpy()
spike = state_snn[:,2,0].detach().numpy()
input = input[:,0].detach().numpy()
decoded = decoded.detach().numpy()

time = np.arange(0,len(input)*time_step,time_step)

fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, sharex=True)
fig.text(0.5, 0.04, 'Time (s)', ha='center')

ax1.set_title("Input Current")
ax1.plot(time,input)
ax1.grid()

ax2.set_title("Membrame potential")
ax2.plot(time,mem_pot)
ax2.axhline(neuron_par["threshold"], color= 'r')
ax2.set_ylim(-80,40)
ax2.grid()

ax3.set_title("Recovery variable")
ax3.plot(time,recovery, color = 'orange')
ax3.grid()

ax4.set_title("Output spikes")
ax4.plot(time,spike) 
ax4.grid()

ax5.set_title("Decoded Output")
ax5.plot(time,decoded,label = "Decoded")
ax5.plot(time[:-1],output_target, label="Target",color = 'r')
ax5.grid()
ax5.legend()

plt.show()


