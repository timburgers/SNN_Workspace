# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
import torch.nn as nn
from models.Izhikevich import LinearIzhikevich
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
from Coding.Decoding import Linear_LI_filter
import numpy as np


class Izhikevich_SNN(nn.Module):
	def __init__(self, param_init, device, config):
		super(Izhikevich_SNN,self).__init__()
		self.input_features = config["NEURONS"]	
		self.neurons = self.input_features
		self.device = device
		izh_time_step = torch.ones(self.neurons)*config["IZHIK_TIME_STEP"]

		self.params_fixed_l1=dict(
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

		self.l1 = LinearIzhikevich(self.input_features,self.neurons,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0),izh_time_step)
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

def initialize_parameters(config):
	np.random.seed(config["RANDOM_INIT_SEED"])
	init_param = {}
	neurons = config["NEURONS"]
	# print("Initialization is set to : ", config["INIT_SETTING"])
	if config["INIT_SETTING"] == "random":
		config_rand = config["INITIAL_PARAMS_RANDOM"]		
		init_param["l1_a"]		= torch.tensor(np.random.uniform(config_rand["a"][0],config_rand["a"][1], size=(neurons))).float()
		init_param["l1_b"]		= torch.tensor(np.random.uniform(config_rand["b"][0],config_rand["b"][1], size=(neurons))).float()
		init_param["l1_c"]		= torch.tensor(np.random.uniform(config_rand["c"][0],config_rand["c"][1], size=(neurons))).float()
		init_param["l1_d"]		= torch.tensor(np.random.uniform(config_rand["d"][0],config_rand["d"][1], size=(neurons))).float()
		init_param["l1_thres"]	= torch.tensor(np.random.uniform(config_rand["threshold"][0],config_rand["threshold"][1], size=(neurons))).float()
		init_param["l1_weights"]= torch.tensor(np.random.uniform(config_rand["weights_1"][0],config_rand["weights_1"][1],size=(neurons,1))).float()
		init_param["l1_v2"]		= torch.tensor(np.random.uniform(config_rand["v2"][0],config_rand["v2"][1], size=(neurons))).float()
		init_param["l1_v1"]		= torch.tensor(np.random.uniform(config_rand["v1"][0],config_rand["v1"][1], size=(neurons))).float()
		init_param["l1_v0"]		= torch.tensor(np.random.uniform(config_rand["v0"][0],config_rand["v0"][1], size=(neurons))).float()
		init_param["l1_tau_u"]	= torch.tensor(np.random.uniform(config_rand["tau_u"][0],config_rand["tau_u"][1], size=(neurons))).float()

		init_param["l2_leak"] 	= torch.tensor(np.random.uniform(config_rand["leak"][0],config_rand["leak"][1], size=(1))).float()
		init_param["l2_weights"]= torch.tensor(np.random.uniform(config_rand["weights_2"][0],config_rand["weights_2"][1],size=(1,neurons))).float()	# random weights [-1,1]
		
		# Set half of the initial input and corresponding output weights to negative values
		if config["HALF_NEGATIVE_WEIGHTS"] == True:
			for idx in range(int(neurons/2)):
				init_param["l1_weights"][idx,:] = init_param["l1_weights"][idx,:] *-1
				init_param["l2_weights"][:, idx] = init_param["l2_weights"] [:, idx]*-1

	# elif init_setting == "manual":
	# 	init_param["l1_a"]		= torch.tensor(l1_init_man["a"]).float()		
	# 	init_param["l1_b"]		= torch.tensor(l1_init_man["b"]).float()		
	# 	init_param["l1_c"]		= torch.tensor(l1_init_man["c"]).float()			
	# 	init_param["l1_d"]		= torch.tensor(l1_init_man["d"]).float()			
	# 	init_param["l1_thres"]	= torch.tensor(l1_init_man["threshold"]).float()
	# 	init_param["l1_weights"]= torch.tensor(l1_init_man["weights"]).float().unsqueeze(1)
		
	# 	init_param["l2_leak"] 	= torch.tensor(l2_init_man["leak"]).float()	
	# 	init_param["l2_weights"]= torch.tensor(l2_init_man["weights"]).float().unsqueeze(0)

	else: 
		print("Init setting not found")
		exit()
	return init_param


def init_param_in_torchga_create_pop(config):
	init_param = np.array([])
	neurons = config["NEURONS"]
	# print("Initialization is set to : ", config["INIT_SETTING"])
	if config["INIT_SETTING"] == "random":
		config_rand = config["INITIAL_PARAMS_RANDOM"]
		w1 = np.random.uniform(config_rand["weights_1"][0],config_rand["weights_1"][1],size=(neurons))
		a = np.random.uniform(config_rand["a"][0],config_rand["a"][1], size=(neurons))
		b= np.random.uniform(config_rand["b"][0],config_rand["b"][1], size=(neurons))
		c = np.random.uniform(config_rand["c"][0],config_rand["c"][1], size=(neurons))
		d = np.random.uniform(config_rand["d"][0],config_rand["d"][1], size=(neurons))
		v2 = np.random.uniform(config_rand["v2"][0],config_rand["v2"][1], size=(neurons))
		v1 = np.random.uniform(config_rand["v1"][0],config_rand["v1"][1], size=(neurons))
		v0 = np.random.uniform(config_rand["v0"][0],config_rand["v0"][1], size=(neurons))
		tau_u = np.random.uniform(config_rand["tau_u"][0],config_rand["tau_u"][1], size=(neurons))
		thres = np.random.uniform(config_rand["threshold"][0],config_rand["threshold"][1], size=(neurons))
		w2 = np.random.uniform(config_rand["weights_2"][0],config_rand["weights_2"][1],size=(neurons))	# random weights [-1,1]
		leak = np.random.uniform(config_rand["leak"][0],config_rand["leak"][1], size=(1))
		
		# Set half of the initial input and corresponding output weights to negative values
		if config["HALF_NEGATIVE_WEIGHTS"] == True:
			for idx in range(int(neurons/2)):
				w1[idx] = w1[idx] *-1
				w2[idx] = w2[idx]*-1

		init_param = np.concatenate((w1,a,b,c,d,v2,v1,v0,tau_u,thres,w2,leak), axis=None)

	else: 
		print("Init setting not found")
		exit()
	return init_param
