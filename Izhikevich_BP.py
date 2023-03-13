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
from Coding.Decoding import sliding_window

# # Proportional
# neuron_par = dict(a = 0.1, b = 0.222, c = -61.6, d = 0, threshold = 30)

# Derivative
neuron_par = dict(a = 0.0105, b = 0.656, c = -55.0, d = 1.92, threshold = 30)

# # Integral
# neuron_par = dict(a = 0.0158, b = 0.139, c = -70.0, d = 1.06, threshold = 30)

weight_syn = 1.
time_step = 0.1 #ms
sim_time = 5000 #ms


#Initialize SNN
SNN_izhik = Izhikevich_SNN(2,neuron_par,weight_syn,time_step)

# Select and simulate input current
seq_len = int(sim_time/time_step)
input = input_cur("webb_advance",seq_len,current_value=5,time_step=time_step,input_features = SNN_izhik.input_features)

# Initialize the states
states = torch.zeros(3,1)
states[1]= -70			# To prevent first spike to happen (set v to -82.65)
states[0]= neuron_par["b"] * states[1]

train_snn(SNN_izhik,train_input)

# Call forward function
output, state = SNN_izhik(input,states)


#Set target for the output decoded data, since input.detach.numpy is of shape (x,1) --> axis of diff is 0
output_target = np.diff(input.detach().numpy(), axis=0)/time_step

decode_time, decode_output = sliding_window(output,time_step,window_size=100,window_step=20)
# decode_output_norm = np.max(output_target)/np.max(decode_output)*decode_output
decode_output_norm = (decode_output-12.5)/34.4


################################# Creating Plots ################################################

# Convert all torch.tensors to numpu element
recovery = state[:,0,0].detach().numpy()
mem_pot = state[:,1,0].detach().numpy()
spike = state[:,2,0].detach().numpy()
input = input.detach().numpy()

t = np.arange(0,sim_time,time_step)

fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, sharex=True)
ax1.set_title("Input Current")
ax1.plot(t,input)
ax1.grid()

ax2.set_title("Membrame potential")
ax2.plot(t,mem_pot)
ax2.axhline(neuron_par["threshold"], color= 'r')
ax2.set_ylim(-80,40)
ax2.grid()

ax3.set_title("Recovery variable")
ax3.plot(t,recovery, color = 'orange')
ax3.grid()

ax4.set_title("Output spikes")
ax4.plot(t,spike) 
ax4.grid()

ax5.set_title("Decoded Output")
ax5.plot(decode_time,decode_output_norm,label = "Decoded")
ax5.plot(t[:-1],output_target, label="Target",color = 'r')
ax5.grid()
ax5.legend()

plt.show()


