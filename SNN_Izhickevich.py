# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
import torch.nn as nn
from models.Izhikevich import LinearIzhikevich
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
from Coding.Decoding import Linear_LI_filter


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

		self.decode_leak = dict(
			leak = torch.ones(1)*0.99
		)

		self.f1 = LinearIzhikevich(self.input_features,self.neurons,self.params_fixed,self.params_learnable,get_spike_fn("ArcTan", 1.0, 10.0))
		self.f2 = Linear_LI_filter(self.neurons,1,{},self.decode_leak,get_spike_fn("ArcTan", 1.0, 10.0))

		# Set the weights in the torch.nn module (Linear() expects the weight matrix in shape (output,input))
		weight_matrix = torch.eye(self.neurons,self.input_features)*weight_syn
		self.f1.ff.weight = torch.nn.parameter.Parameter(weight_matrix)

		weight_matrix = torch.ones(1,self.neurons)*weight_syn
		self.f2.ff.weight = torch.nn.parameter.Parameter(weight_matrix)


	def forward(self,input_batch,state):

		seq_length, n_inputs = input_batch.size()
		outputs = []
		states = []
		decoded = []
		decode_mempot = torch.tensor([0])
		for timestep in range(seq_length):
			input = input_batch[timestep,:]
			state,output = self.f1(state,input)
			decode_mempot, _ = self.f2(decode_mempot,output)
			outputs += [output]
			states += [state]
			decoded += [decode_mempot]
		
		outputs = torch.stack(outputs)
		states = torch.stack(states)
		decoded = torch.stack(decoded)
		
		return outputs, states, decoded

