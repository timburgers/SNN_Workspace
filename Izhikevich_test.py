# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
import torch.nn as nn
from models.Izhikevich import LinearIzhikevich
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
import matplotlib.pyplot as plt
import numpy as np
from Coding.Input_current_sim import input_cur

# Proportional
a = 0.1
b = 0.222
c = -61.6
d = 0

# # Derivative
# a = 0.0105
# b = 0.656
# c = -55.0
# d = 1.92

# #integral
# a = 0.0158
# b = 0.139
# c = -70.0
# d = -1.06

threshold = 30
weight_syn = 1.

time_step = 0.1 #ms
sim_time = 2500 #ms

class Izhikevich_SNN(nn.Module):
	def __init__(self):
		super(Izhikevich_SNN,self).__init__()
		self.input_f1 = 1
		self.output_f1 = 1

		params_fixed=dict(
			a = torch.tensor(a),
			b = torch.tensor(b),
			c = torch.tensor(c),
			d = torch.tensor(d),
            thresh= torch.tensor(threshold),
	    	time_step = torch.tensor(time_step)
		)
		params_learnable={}

		self.f1 = LinearIzhikevich(self.input_f1,self.output_f1,params_fixed,params_learnable,get_spike_fn("ArcTan", 1.0, 10.0))
		
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


# Select and simulate input current
seq_len = int(sim_time/time_step)
input = input_cur("webb",seq_len,current_value=5,time_step=time_step)


#Initialize neuron
neuron = Izhikevich_SNN()
states = torch.zeros(3,1)
states[1]= -70			# To prevent first spike to happen (set v to -82.65)
states[0]= b * states[1]



# Call forward function
output, state = neuron(input,states)
print(output)
print(output.shape)
recovery = state[:,0,0].detach().numpy()
mem_pot = state[:,1,0].detach().numpy()
spike = state[:,2,0].detach().numpy()
input = input.detach().numpy()
# output = output.detach().numpy()
t = np.arange(0,sim_time,time_step)

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4, sharex=True)
ax1.set_title("Input Current")
ax1.plot(t,input)

ax2.set_title("Membrame potential")
ax2.plot(t,mem_pot)
ax2.axhline(threshold, color= 'r')
ax2.set_ylim(-80,40)


ax3.set_title("Recovery variable")
ax3.plot(t,recovery, color = 'orange')
ax3.set_ylim(-100,100)

ax4.set_title("Output spikes")
ax4.plot(t,spike) 
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
plt.show()


