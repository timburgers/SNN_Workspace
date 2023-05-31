import torch
import torch.nn as nn
from spiking.torch.layers.linear import RecurrentLinearLIF
from models.spiking.spiking.torch.utils.surrogates import get_spike_fn
from Coding.Decoding import Linear_LI_filter
import numpy as np


class LIF_SNN(nn.Module):
	def __init__(self, param_init, device, config):
		super(LIF_SNN,self).__init__()
		self.input_features = config["NEURONS"]	
		self.neurons = self.input_features
		self.device = device

		if param_init == None:
			param_init = init_empty(config)

		self.params_fixed_l1=dict(thresh = param_init["l1_thres"])
		self.params_learnable_l1=dict(leak_i = param_init["l1_leak_i"], leak_v = param_init["l1_leak_v"])

		self.params_fixed_l2=dict(leak = param_init["l2_leak"])
		self.params_learnable_l2 = dict()

		self.l1 = RecurrentLinearLIF(self.input_features,self.neurons,self.params_fixed_l1,self.params_learnable_l1,get_spike_fn("ArcTan", 1.0, 20.0))
		self.l2 = Linear_LI_filter(self.neurons,1,self.params_fixed_l2,self.params_learnable_l2,None)

		# Set the weights in the torch.nn module (Linear() expects the weight matrix in shape (output,input))
		self.l1.ff.weight = torch.nn.parameter.Parameter(param_init["l1_weights"])
		self.l1.ff.bias = torch.nn.parameter.Parameter(param_init["l1_bias"])
		self.l1.rec.weight = torch.nn.parameter.Parameter(param_init["l1_weights_rec"])
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
		
		spikes = torch.stack(outputs)
		states_snn = torch.stack(states_snn)
		decoded = torch.stack(decoded)
		
		return spikes, states_snn, decoded









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

# def initialize_parameters(config):
# 	print("initialize function used")
# 	np.random.seed(config["RANDOM_INIT_SEED"])
# 	init_param = {}
# 	neurons = config["NEURONS"]
# 	# print("Initialization is set to : ", config["INIT_SETTING"])
# 	if config["INIT_SETTING"] == "random":
# 		config_rand = config["INITIAL_PARAMS_RANDOM"]		
# 		init_param["l1_thres"]	= torch.tensor(np.random.uniform(config_rand["threshold"][0],config_rand["threshold"][1], size=(neurons))).float()
# 		init_param["l1_weights"]= torch.tensor(np.random.uniform(config_rand["weights_1"][0],config_rand["weights_1"][1],size=(neurons,1))).float()
# 		init_param["l1_leak_i"]	= torch.tensor(np.random.uniform(config_rand["leak_i"][0],config_rand["leak_i"][1], size=(neurons))).float()
# 		init_param["l1_leak_v"]	= torch.tensor(np.random.uniform(config_rand["leak_v"][0],config_rand["leak_v"][1], size=(neurons))).float()
# 		init_param["l1_weights_rec"]= torch.tensor(np.random.uniform(config_rand["weights_1"][0],config_rand["weights_1"][1],size=(neurons,1))).float()


# 		init_param["l2_leak"] 	= torch.tensor(np.random.uniform(config_rand["leak"][0],config_rand["leak"][1], size=(1))).float()
# 		init_param["l2_weights"]= torch.tensor(np.random.uniform(config_rand["weights_2"][0],config_rand["weights_2"][1],size=(1,neurons))).float()	# random weights [-1,1]
		
# 		# Set half of the initial input and corresponding output weights to negative values
# 		if config["HALF_NEGATIVE_WEIGHTS"] == True:
# 			for idx in range(int(neurons/2)):
# 				init_param["l1_weights"][idx,:] = init_param["l1_weights"][idx,:] *-1
# 				init_param["l2_weights"][:, idx] = init_param["l2_weights"] [:, idx]*-1

# 	else: 
# 		print("Init setting not found")
# 		exit()
# 	return init_param


# def init_param_in_torchga_create_pop(config):
# 	init_param = np.array([])
# 	neurons = config["NEURONS"]
# 	# print("Initialization is set to : ", config["INIT_SETTING"])
# 	if config["INIT_SETTING"] == "random":
# 		config_rand = config["INITIAL_PARAMS_RANDOM"]
# 		w1 = np.random.uniform(config_rand["weights_1"][0],config_rand["weights_1"][1],size=(neurons))
# 		leak_i = 0
# 		thres = np.random.uniform(config_rand["threshold"][0],config_rand["threshold"][1], size=(neurons))
# 		w2 = np.random.uniform(config_rand["weights_2"][0],config_rand["weights_2"][1],size=(neurons))	# random weights [-1,1]
# 		leak = np.random.uniform(config_rand["leak"][0],config_rand["leak"][1], size=(1))
		
# 		# Set half of the initial input and corresponding output weights to negative values
# 		if config["HALF_NEGATIVE_WEIGHTS"] == True:
# 			for idx in range(int(neurons/2)):
# 				w1[idx] = w1[idx] *-1
# 				w2[idx] = w2[idx]*-1

# 		init_param = np.concatenate((w1,a,b,c,d,v2,v1,v0,tau_u,thres,w2,leak), axis=None)

# 	else: 
# 		print("Init setting not found")
# 		exit()
# 	return init_param


def init_empty(config):

	init_param = {}
	neurons = config["NEURONS"]

	init_param["l1_thres"]	= torch.ones(neurons).float()
	init_param["l1_leak_i"]	= torch.ones(neurons).float()
	init_param["l1_leak_v"]	= torch.ones(neurons).float()
	init_param["l1_weights"]= torch.ones((neurons,1)).float()
	init_param["l1_weights_rec"]= torch.ones((neurons,neurons)).float()
	init_param["l1_bias"]	= torch.ones(neurons).float()

	init_param["l2_leak"] 	= torch.ones(1).float()
	init_param["l2_weights"]= torch.ones((1,neurons)).float()	# random weights [-1,1]
		
	return init_param
