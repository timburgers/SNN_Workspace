# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
import torch.nn as nn
from models.Izhikevich import LinearIzhikevich
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
from Coding.Encoding import input_cur
import matplotlib.pyplot as plt
import numpy as np


class Izhikevich_SNN(nn.Module):
	def __init__(self,input_feat, neuron_params, weight_syn, time_step):
		super(Izhikevich_SNN,self).__init__()
		self.input_features = input_feat
		self.neurons = self.input_features

		self.params_fixed=dict(
			thresh = torch.ones(self.neurons)*neuron_params["threshold"],
			time_step = torch.ones(self.neurons)*time_step
		)
		self.params_learnable=dict(
			a = torch.ones(self.neurons)*neuron_params["a"],
			b = torch.ones(self.neurons)*neuron_params["b"],
			c = torch.ones(self.neurons)*neuron_params["c"],
			d = torch.ones(self.neurons)*neuron_params["d"]
		)


		self.f1 = LinearIzhikevich(self.input_features,self.neurons,self.params_fixed,self.params_learnable,get_spike_fn("ArcTan", 1.0, 10.0))

		# Set the weights in the torch.nn module (Linear() expects the weight matrix in shape (output,input))
		weight_matrix = torch.eye(self.neurons,self.input_features)*weight_syn
		self.f1.ff.weight = torch.nn.parameter.Parameter(weight_matrix)


	def forward(self,input_batch,state):

		seq_length, n_inputs = input_batch.size()
		outputs = []
		states = []
		for timestep in range(seq_length):
			input = input_batch[timestep,:]
			state,output = self.f1(state,input)
			outputs += [output]
			states += [state]
		
		outputs = torch.stack(outputs)
		states = torch.stack(states)
		return outputs, states

time_step = 2
step_size = 1/20

input = input_cur("step",2500,time_step,step_size,1, step_percentage=0.3)
# neuron_par = dict(a = 0.01, b = 0.2, c = -65.0, d = 8, threshold = 30)
# neuron_par = dict(a = 0.1, b = 0.2, c = -65.0, d = 2, threshold = 30) #FS
# neuron_par = dict(a = 0.01, b = 0.2, c = -65.0, d = 8, threshold = 30) 	# F spike frequency adaption
neuron_par = dict(a = 0.02, b = -0.1, c = -55.0, d = 6, threshold = 30) 	# G class 1 excitable
# neuron_par = dict(a = 0.2, b = 0.26, c = -65.0, d = 0, threshold = 30) 	# H class 2 excitable
# neuron_par = dict(a = 0.02, b = 1, c = -55.0, d = 4, threshold = 30) 		# R Accomodation
# neuron_par = dict(a = -0.02, b = -1, c = -60.0, d = 8, threshold = 30) 	# S Inhibition-induced spiking


states = torch.zeros(3,1)
states[1]= -70			# To prevent first spike to happen (set v to -82.65)
states[0]= neuron_par["b"]*states[1]

neuron_F_Izhickevich = Izhikevich_SNN(1,neuron_par,1,time_step)
spike_snn, state_snn = neuron_F_Izhickevich(input,states)





# Convert all torch.tensors to numpu element
recovery = state_snn[:,0,0].detach().numpy()
mem_pot = state_snn[:,1,0].detach().numpy()
spike = state_snn[:,2,0].detach().numpy()
input = input[:,0].detach().numpy()

time = np.arange(0,len(input)*time_step,time_step)

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4, sharex=True)
fig.text(0.5, 0.04, 'Time (ms)', ha='center')

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


plt.show()


