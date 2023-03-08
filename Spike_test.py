import torch
import torch.nn as nn
from spiking.torch.layers.linear import LinearLIF
from spiking.torch.utils.surrogates import get_spike_fn
import matplotlib.pyplot as plt
import numpy as np
import random


tau_syn = 7
tau_mem = 10
threshold = 1
weight_syn = 0.5




class SNN_test(nn.Module):
	def __init__(self):
		super(SNN_test,self).__init__()
		self.input_f1 = 1
		self.output_f1 = 1

		leakage_syn= torch.tensor(1-1/tau_syn)
		leakage_mem= torch.tensor(1-1/tau_mem)
		thresh_v = torch.tensor(threshold)

		params_fixed=dict(
			leak_i= leakage_syn,
            leak_v =leakage_mem,
            thresh= thresh_v
		)
		params_learnable={}

		self.f1 = LinearLIF(self.input_f1,self.output_f1,params_fixed,params_learnable,get_spike_fn("ArcTan", 1.0, 10.0))

		# Set the weights in the torch.nn module
		weight = torch.tensor([weight_syn])
		self.f1.ff.weight = torch.nn.parameter.Parameter(weight)


	def forward(self,x,state):

		seq_length, n_inputs = x.size()
		outputs = []
		states = []
		for timestep in range(seq_length):
			input = x[timestep,:]
			state,output = self.f1(state,input)
			outputs += [output]
			states += [state]
		outputs = torch.stack(outputs)
		states = torch.stack(states)
		
		return outputs, states

seq_len = 1000
input_cur = 0.03

# Generate random spike pattern
percentage_of_spikes = 0.02
spike_list = [0]*int(seq_len*(1-percentage_of_spikes))+[0.1]*int(seq_len*percentage_of_spikes)
random.shuffle(spike_list)
spike_list = torch.FloatTensor(spike_list)
spike_input = torch.reshape(spike_list,(seq_len,1))

# Set constant input current
const_input = torch.zeros(seq_len,1)
for i in range(seq_len):
	# input[i,0]=0.00001*i
	const_input[i,0]=input_cur

# Set constant input current
var_input = torch.zeros(seq_len,1)
for i in range(seq_len):
	# input[i,0]=0.00001*i
	if i < (seq_len/2): var_input[i,0]=input_cur*i
	else: var_input[i,0]=input_cur*(seq_len - i)


# Select input type
input = const_input

#Initialize neuron
neuron = SNN_test()
states = torch.zeros(3,1)

# Call forward function
output, state = neuron(input,states)


current = state[:,0,0].detach().numpy()
mem_pot = state[:,1,0].detach().numpy()
spike = state[:,2,0].detach().numpy()
input = input.detach().numpy()
# output = output.detach().numpy()
t = np.arange(0,seq_len)

fig,(ax1,ax2,ax3) = plt.subplots(3, sharex=True)
ax1.set_title("Input Current")
ax1.plot(t,current, label="Current state")
ax1.plot(t,input, label ="Current in")
ax1.legend()

ax2.set_title("Membrame potential")
ax2.plot(t,mem_pot)

ax3.set_title("Output spikes")
ax3.plot(t,spike) 
ax1.grid()
ax2.grid()
ax3.grid()
plt.show()