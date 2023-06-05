import torch
import torch.nn as nn
from spiking.torch.layers.linear import RecurrentLinearLIF, LinearLIF
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
from models.Leaky_integrator import Linear_LI_filter
import numpy as np


class BASE_LIF_SNN(nn.Module):
	def __init__(self, neurons):
		super(BASE_LIF_SNN,self).__init__()
		self.input_features = neurons
		self.neurons = neurons
		self.params_fixed_l1= dict()
		self.params_fixed_l2 = dict()

	def forward(self,input_batch,state_snn, state_LI):

		batch_size, seq_length, n_inputs = input_batch.size()
		# Different pass if the for loop is outside of the forward pass aka, seq_len is one
		if seq_length ==1:
			input = input_batch[:,0,:]
			states_snn,spikes = self.l1(state_snn,input)
			decoded, _ = self.l2(state_LI,spikes)
		
		# if forward pass is inside this pass, aka seq_len is larger than 1
		else:
			outputs, states_snn, decoded = [],[],[]
			for timestep in range(seq_length):
				input = input_batch[:,timestep,:]
				state_snn,output = self.l1(state_snn,input)
				state_LI, _ = self.l2(state_LI,output)
				outputs += [output]
				states_snn += [state_snn]
				decoded += [state_LI]
			spikes = torch.stack(outputs)
			states_snn = torch.stack(states_snn)
			decoded = torch.stack(decoded)
		
		return spikes, states_snn, decoded

class LIF_SNN(BASE_LIF_SNN):
	def __init__(self, param_init, neurons,layer_settings):
		super(LIF_SNN,self).__init__(neurons)


		if param_init == None: param_init = init_empty(neurons,layer_settings)

		self.params_learnable_l1=dict(leak_i = param_init["l1_leak_i"], leak_v = param_init["l1_leak_v"],thresh = param_init["l1_thres"])
		self.params_learnable_l2 = dict(leak = param_init["l2_leak"])

		# Either use the recurrent linear layer or the baselinear layer
		if layer_settings["l1"]["recurrent"]== True:
			self.l1 = RecurrentLinearLIF(self.input_features,self.neurons,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0), layer_settings["l1"])
			self.l1.rec.weight = torch.nn.parameter.Parameter(param_init["l1_weights_rec"])
		else: 
			self.l1 = LinearLIF(self.input_features,self.neurons,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0), layer_settings["l1"])

		self.l2 = Linear_LI_filter(self.neurons,1,self.params_fixed_l2,self.params_learnable_l2,None, layer_settings["l2"])

		# Set the weights in the torch.nn module (Linear() expects the weight matrix in shape (output,input))
		self.l1.ff.weight = torch.nn.parameter.Parameter(param_init["l1_weights"])
		self.l1.ff.bias = torch.nn.parameter.Parameter(param_init["l1_bias"])
		
		self.l2.ff.weight = torch.nn.parameter.Parameter(param_init["l2_weights"])

def init_empty(neurons,layer_set):
	init_param = {}

	init_param["l1_thres"]	= torch.ones(neurons).float()
	init_param["l1_leak_v"]	= torch.ones(neurons).float()
	init_param["l2_leak"] 	= torch.ones(1).float()

	if layer_set["l1"]["recurrent"] == True:
		init_param["l1_weights_rec"]= torch.ones(neurons,neurons).float()

	if layer_set["l1"]["shared_weight_and_bias"] == True: 
		init_param["l1_weights"]= torch.ones(int(neurons/2),1).float() 	#NOTE: shape must be (neurons,1)
		init_param["l1_bias"]	= torch.ones(int(neurons/2)).float()
	else:                                        
		init_param["l1_weights"]= torch.ones(neurons,1).float()			#NOTE: shape must be (neurons,1)
		init_param["l1_bias"]	= torch.ones(neurons).float()
	
	if layer_set["l1"]["shared_leak_i"] == True:
		init_param["l1_leak_i"] = torch.ones(int(neurons/2))
	else: 
		init_param["l1_leak_i"]	= torch.ones(neurons).float() 


	if layer_set["l2"]["shared_weight_and_bias"] == True:
		init_param["l2_weights"]= torch.ones(1,int(neurons/2)).float()	#NOTE: shape must be (1, neurons)
	else:
		init_param["l2_weights"]= torch.ones(1,neurons).float()			#NOTE: shape must be (1, neurons)
	
	
	return init_param
