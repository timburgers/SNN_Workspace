# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
import torch.nn as nn
from models.Izhikevich import LinearIzhikevich
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn


class Izhikevich_SNN(nn.Module):
	def __init__(self,input_feat, neuron_params, weight_syn, time_step):
		super(Izhikevich_SNN,self).__init__()
		self.input_features = input_feat
		self.hidden_neurons = self.input_features

		params_fixed=dict(
            thresh= torch.tensor(neuron_params["threshold"]),
	    	time_step = torch.tensor(time_step)
		)
		params_learnable=dict(
			a = torch.tensor(neuron_params["a"]),
			b = torch.tensor(neuron_params["b"]),
			c = torch.tensor(neuron_params["c"]),
			d = torch.tensor(neuron_params["d"])
		)

		self.f1 = LinearIzhikevich(self.input_features,self.hidden_neurons,params_fixed,params_learnable,get_spike_fn("ArcTan", 1.0, 10.0))
		
		# Set the weights in the torch.nn module (Linear() expects the weight matrix in shape (output,input))
		weight_matrix = torch.eye(self.hidden_neurons,self.input_features)*weight_syn
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

