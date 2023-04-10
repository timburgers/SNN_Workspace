# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
import torch.nn as nn
from models.Izhikevich import LinearIzhikevich
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
from Coding.Decoding import Linear_LI_filter
import numpy as np


# # Proportional
# neuron_par = dict(a = 0.1, b = 0.222, c = -61.6, d = 0, threshold = 30)

# Derivative
# neuron_par = dict(a = 0.0105, b = 0.656, c = -55.0, d = 1.92, threshold = 30)
# neuron_par = dict(a = 0.0000105, b = 0.000656, c = -0.055, d = 0.00192, threshold = 0.030)

# # Integral
# neuron_par = dict(a = 0.0158, b = 0.139, c = -70.0, d = 1.06, threshold = 30)

class Izhikevich_SNN(nn.Module):
	def __init__(self,input_feat, param_init, time_step , device):
		super(Izhikevich_SNN,self).__init__()
		self.input_features = input_feat
		self.neurons = self.input_features
		self.device = device
		

		self.params_fixed_l1=dict(
			time_step = torch.ones(self.neurons)*time_step,
			thresh = param_init["l1_thres"]

		)
		self.params_learnable_l1=dict(
			a = param_init["l1_a"],
			b = param_init["l1_b"],
			c = param_init["l1_c"],
			d = param_init["l1_d"],
			v2 = param_init["l1_v2"],
			v1 = param_init["l1_v1"],
			v0 = param_init["l1_v0"],
			tau_u = param_init["l1_tau_u"],

		)

		self.params_fixed_l2=dict(
			leak = param_init["l2_leak"]
		)
		self.params_learnable_l2 = dict(
		)

		self.l1 = LinearIzhikevich(self.input_features,self.neurons,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0))
		self.l2 = Linear_LI_filter(self.neurons,1,self.params_fixed_l2,self.params_learnable_l2,None)

		# Set the weights in the torch.nn module (Linear() expects the weight matrix in shape (output,input))
		self.l1.ff.weight = torch.nn.parameter.Parameter(param_init["l1_weights"])
		self.l2.ff.weight = torch.nn.parameter.Parameter(param_init["l2_weights"])

	def forward(self,input_batch,state_snn, state_LI):

		batch_size, seq_length, n_inputs = input_batch.size()
		outputs, states_snn, decoded = [],[],[]
		

		for timestep in range(seq_length):
			input = input_batch[:,timestep,:]
			state_snn,output = self.l1(state_snn,input)
			state_LI, _ = self.l2(state_LI,output)

			outputs += [output]
			states_snn += [state_snn]
			decoded += [state_LI]
		
		outputs = torch.stack(outputs)
		states_snn = torch.stack(states_snn)
		decoded = torch.stack(decoded)
		
		return outputs, states_snn, decoded


def list_learnable_parameters(network, show_results):
	if show_results == True:
		print ("Print the parameters of the network \n")
	learn_param_dic = {}
	for name, param in network.named_parameters():
		if param.requires_grad:
			if show_results == True:
				print(name, param.data)	
			else:
				learn_param_dic[name] =  param.data.cpu().detach().numpy()
	if show_results == True:
		print("\n")
		return	
	else: return learn_param_dic