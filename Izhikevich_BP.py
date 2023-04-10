# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
import torch.nn as nn
from SNN_Izhickevich import Izhikevich_SNN, list_learnable_parameters
import matplotlib.pyplot as plt
import numpy as np
from Coding.Encoding import input_cur
from Coding.Decoding import sliding_window, Linear_LI_filter
from Backprop_SNN import train_SNN
import pandas as pd
import os
from dataset_creation.pytorch_dataset import Dataset_derivative
import wandb



### Training + Network parameters
NEURONS = 10
LEARNING_RATE_DEFAULT = 0
EPOCHS = 50
NUM_BATCHES_PER_DATAFILE = 20
PERCENT_TRAIN = 0.40
BATCH_SIZE = 128
TIME_STEP_TEST = 0.002
DATASET_DIR = "Sim_data/derivative/dt0.002_norm_neg"
INPUT_COLUMN_DATAFILE = [1]
LABEL_COLUMN_DATAFILE = [1]
RANDOM_INIT_SEED = 1 #np.random.randint(0,1000000)
LR_L1_NEURON = 0.01
LR_L1_WEIGHTS= 0.01
LR_L2_NEURON = 0
LR_L2_WEIGHTS= 0.01


INIT_SETTING = "random"		#random or manual
WANDB_LOG = True
SHOW_PLOTS = True

# Set the training bounds for the trainable parameters in the form [min,max]
PARAMETER_BOUNDS = dict({
	# "l1.neuron.a": 		[0,0.5],
	# "l1.neuron.b": 		[0, 0.35],
	# "l1.neuron.c": 		[-100, 0],
	# "l1.neuron.d": 		[0, 10],
	"l1.ff.weight": 	[None,None],
	# "l2.neuron.leak":	[0,1],
	"l2.ff.weight": 	[-1,1]    
})
IZHIK_TIME_STEP = 0.02
lr_parameters = dict({'l1_neuron_lr': LR_L1_NEURON,
				'l1_weights_lr': LR_L1_WEIGHTS,
				'l2_neuron_lr':  LR_L2_NEURON,
				'l2_weights_lr': LR_L2_WEIGHTS})


### Random init
L1_INIT_RANDOM = {	"a_min": 0.019, 	"a_max": 0.021, 
					"b_min": 0.41, 		"b_max": 0.43, 
					"c_min": -56, 		"c_max": -54, 
					"d_min": 0, 		"d_max": 0, 
					"thresh_min":30,	"thresh_max":30,
					"v2_min": 0.1,		"v2_max":0.15,
					"v1_min": 7,		"v1_max":7.5,
					"v0_min": 42,		"v0_max":48,
					"tau_u_min": 4,	"tau_u_max":6,
					"weight_min":50, "weight_max":100}
L2_INIT_RANDOM = {	"leak_min":0.90, 	"leak_max":0.90,
		       	  	"weight_min":0.15, 	"weight_max":0.15}
HALF_NEGATIVE_WEIGHTS = True



### Manual init
L1_INIT_MANUAL = {	"a"			:[0.01	,0.02],
					"b"			:[0.2	,0.4],
					"c"			:[-40	,-40],
					"d"			:[0		,0],
					"threshold"	:[20	,30],
					"weights"	:[0.5	,0.5]}
L2_INIT_MANUAL = {	"leak"		:[0.9999],
		       	  	"weights"	:[-0.2	,0.5]}



def test_SNN(network,train_loss,run):
	### Test network (on a single file)
	input_test = pd.read_csv(DATASET_DIR+ "/dataset_3.csv", usecols=INPUT_COLUMN_DATAFILE, header=None, skiprows=[0])
	input_test = torch.tensor(input_test.values).float() 	# convert from pandas df to torch tensor and floats 
	# input_test = torch.tile(input_test,(1,NEURONS))		# convert from (seq_len,1) to (seq_len ,features)
	input_test = input_test.unsqueeze(0)					# shape from (seq_len ,features) to (1, seq, feature)										

	### Initialize the states(u, v, s)
	states_snn = torch.zeros(3,NEURONS, device = torch.device("cpu"))
	state_LI = torch.zeros(1, device = torch.device("cpu"))


	### Call forward function
	network = network.to("cpu")
	spike_test, state_test, decoded_test = network(input_test,states_snn,state_LI)

	### Set target for the output decoded data, since input.detach.numpy is of shape (x,1) --> axis of diff is 0
	target_test = pd.read_csv(DATASET_DIR + "/dataset_3.csv", usecols=LABEL_COLUMN_DATAFILE, header=None, skiprows=[0]).to_numpy()


	### Convert all torch.tensors to numpy element. snn shape = (seq_len, states, batch, features)
	spike = state_test[:,2,0,:].detach().numpy()
	sw_time, sw_spikes  = sliding_window(spike,TIME_STEP_TEST,0.01,TIME_STEP_TEST)

	input = input_test[0,:,0].detach().numpy()
	output = decoded_test[:,0,0].detach().numpy()
	time = np.arange(0,len(input)*TIME_STEP_TEST,TIME_STEP_TEST)
	

	create_wandb_summary_table(network,run,spike)

	fig,(ax1,ax2,ax3) = plt.subplots(3, sharex=True)
	fig.text(0.5, 0.04, 'Time (s)', ha='center')

	ax1.set_title("Input Current")
	ax1.plot(time,input)
	ax1.grid()

	ax2.set_title("Output spikes count (every 0.01 sec)")
	for input_neuron in range(np.size(sw_spikes,axis=1)):
		if input_neuron < int(NEURONS/2):
			color = 'r'
		else: color = 'g'
		ax2.plot(sw_time,sw_spikes[:,input_neuron],label=str(input_neuron), color = color)
	ax2.grid()
	ax2.legend()

	ax3.set_title("Decoded Output")
	ax3.plot(time,output,label = "Decoded")
	ax3.plot(time,target_test, label="Target",color = 'r')
	ax3.grid()
	ax3.legend()
	if SHOW_PLOTS ==True: plt.show(block=False)
	
	fig2,ax2 = plt.subplots()
	epochs_list = np.arange(1,EPOCHS+1)
	ax2.plot(epochs_list,train_loss)
	ax2.set_title("Loss over epochs")
	ax2.set_xlabel("Epochs [-]")
	ax2.set_ylabel("Loss")
	ax2.grid()
	if SHOW_PLOTS ==True: plt.show()

	### Log the first plot to wandb
	run.log({"Test trial": fig})



def initialize_parameters(init_setting, l1_init_rand, l2_init_rand, l1_init_man, l2_init_man, half_negative, random_init_seed):
	np.random.seed(random_init_seed)
	init_param = {}
	print("Initialization is set to : ", init_setting)
	if init_setting == "random":		
		init_param["l1_a"]		= torch.tensor(np.random.uniform(l1_init_rand["a_min"],l1_init_rand["a_max"], size=(NEURONS))).float()
		init_param["l1_b"]		= torch.tensor(np.random.uniform(l1_init_rand["b_min"],l1_init_rand["b_max"], size=(NEURONS))).float()
		init_param["l1_c"]		= torch.tensor(np.random.uniform(l1_init_rand["c_min"],l1_init_rand["c_max"], size=(NEURONS))).float()
		init_param["l1_d"]		= torch.tensor(np.random.uniform(l1_init_rand["d_min"],l1_init_rand["d_max"], size=(NEURONS))).float()
		init_param["l1_thres"]	= torch.tensor(np.random.uniform(l1_init_rand["thresh_min"],l1_init_rand["thresh_max"], size=(NEURONS))).float()
		init_param["l1_weights"]= torch.tensor(np.random.uniform(l1_init_rand["weight_min"],l1_init_rand["weight_max"],size=(NEURONS,1))).float()
		init_param["l1_v2"]		= torch.tensor(np.random.uniform(l1_init_rand["v2_min"],l1_init_rand["v2_max"], size=(NEURONS))).float()
		init_param["l1_v1"]		= torch.tensor(np.random.uniform(l1_init_rand["v1_min"],l1_init_rand["v1_max"], size=(NEURONS))).float()
		init_param["l1_v0"]		= torch.tensor(np.random.uniform(l1_init_rand["v0_min"],l1_init_rand["v0_max"], size=(NEURONS))).float()
		init_param["l1_tau_u"]	= torch.tensor(np.random.uniform(l1_init_rand["tau_u_min"],l1_init_rand["tau_u_max"], size=(NEURONS))).float()

		init_param["l2_leak"] 	= torch.tensor(np.random.uniform(l2_init_rand["leak_min"],l2_init_rand["leak_max"], size=(1))).float()
		init_param["l2_weights"]= torch.tensor(np.random.uniform(l2_init_rand["weight_min"],l2_init_rand["weight_max"],size=(1,NEURONS))).float()	# random weights [-1,1]
		
		# Set half of the initial input and corresponding output weights to negative values
		if half_negative == True:
			for idx in range(int(NEURONS/2)):
				init_param["l1_weights"][idx,:] = init_param["l1_weights"][idx,:] *-1
				init_param["l2_weights"][:, idx] = init_param["l2_weights"] [:, idx]*-1

	elif init_setting == "manual":
		init_param["l1_a"]		= torch.tensor(l1_init_man["a"]).float()		
		init_param["l1_b"]		= torch.tensor(l1_init_man["b"]).float()		
		init_param["l1_c"]		= torch.tensor(l1_init_man["c"]).float()			
		init_param["l1_d"]		= torch.tensor(l1_init_man["d"]).float()			
		init_param["l1_thres"]	= torch.tensor(l1_init_man["threshold"]).float()
		init_param["l1_weights"]= torch.tensor(l1_init_man["weights"]).float().unsqueeze(1)
		
		init_param["l2_leak"] 	= torch.tensor(l2_init_man["leak"]).float()	
		init_param["l2_weights"]= torch.tensor(l2_init_man["weights"]).float().unsqueeze(0)
	
	else: 
		print("Init setting not found")
		exit()
	return init_param

def create_wandb_summary_table(network,run,spike_train):
	### print total spike count per neuron
	spike_count = []
	for neuron in range(NEURONS):
		print("Spike count neuron ", neuron, " = ", np.sum(spike_train[:,neuron]) )
		spike_count.append(float(np.sum(spike_train[:,neuron])))

	data = []
	names = []
	for name, param in network.named_parameters():
		if param.requires_grad:
				names.append(name)
				# ravel makes to converts the multi dim ndarrays to a single dimension
				data.append(list(param.detach().numpy().ravel()))
	names.append("Test Spike Count")
	data.append(spike_count)
	data = np.array(data).T.tolist()


	summary_table = wandb.Table(rows=np.arange(0,NEURONS).tolist(), columns=names,data=data)
	run.log({"Trained parameters overview": summary_table})
#

def difference_learning_params(before, after):
	print("\n Difference in network parameters \n")
	for name in after:
		print(name)
		print((after[name]-before[name])) 

	print("\n \n")



def set_config_wandb(network):
	if LABEL_COLUMN_DATAFILE[0] ==1:
		TYPE_NN = "Proportional"
	if LABEL_COLUMN_DATAFILE[0] == 2:
		TYPE_NN = "Derivative"

	# Send the constants/hyper parameters to WANDB
	wandb.config.update({"lr_l1_neuron": LR_L1_NEURON,
						"lr_l1_weights": LR_L1_WEIGHTS,
						"lr_l2_neuron": LR_L2_NEURON,
						"lr_l2_weights": LR_L2_WEIGHTS, 
						"Epochs": EPOCHS,
						"Batch size": BATCH_SIZE,
						"Percentage train":PERCENT_TRAIN,
						"Training data": DATASET_DIR,
						"Neurons": network.neurons, 
						"NN model": TYPE_NN,
						"Weight init seed": RANDOM_INIT_SEED,
						"half weights negative":HALF_NEGATIVE_WEIGHTS})

def print_network_training_parameters():
	print("Parameter Bounds")
	print(PARAMETER_BOUNDS)

	print("\n Learning Rates for the parameters")
	print(lr_parameters)
	if INIT_SETTING == "random":
		print("\n Inital RANDOM values for the parameters")
		print(L1_INIT_RANDOM)	
		print(L2_INIT_RANDOM)	

	if INIT_SETTING == "manual":
		print("\n Inital MANUAL values for the parameters")
		print(L1_INIT_MANUAL)
		print(L2_INIT_MANUAL)
	


def main():
	### Initialize SNN + Dataset
	param_init = initialize_parameters(INIT_SETTING,L1_INIT_RANDOM,L2_INIT_RANDOM,L1_INIT_MANUAL,L2_INIT_MANUAL,HALF_NEGATIVE_WEIGHTS,RANDOM_INIT_SEED)
	SNN_izhik = Izhikevich_SNN(NEURONS,param_init,IZHIK_TIME_STEP, device).to(device)
	dataset = Dataset_derivative(DATASET_DIR, NUM_BATCHES_PER_DATAFILE, NEURONS, INPUT_COLUMN_DATAFILE, LABEL_COLUMN_DATAFILE)  

	### Set weight & biases
	run = wandb.init(project= "SNN_Izhikevich_P", mode=WANDB_MODE, reinit=True)	
	set_config_wandb(SNN_izhik)
	wandb.watch(SNN_izhik, log="parameters", log_freq=1)

	### Train the network
	# param_before_training = list_learnable_parameters(SNN_izhik, show_results=True)
	param_before_training = list_learnable_parameters(SNN_izhik, show_results=False)
	trained_network, train_loss = train_SNN(SNN_izhik, dataset, EPOCHS, PERCENT_TRAIN, LEARNING_RATE_DEFAULT, BATCH_SIZE,RANDOM_INIT_SEED, device, PARAMETER_BOUNDS,lr_parameters)
	param_after_training = list_learnable_parameters(trained_network, show_results=False)
	
	### Calculate the difference between start and final learnable parameters
	difference_learning_params(param_before_training,param_after_training)
	param_after_training = list_learnable_parameters(trained_network, show_results=True)

	### Test the network
	test_SNN(trained_network, train_loss,run)

	print_network_training_parameters()



if __name__ == '__main__':
	os.system('cls')

	if WANDB_LOG==True:
		WANDB_MODE = "online"	# "online", "offline" or "disabled"
	else:
		WANDB_MODE = "disabled"
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	device = "cpu"
	print("Device = ", device)

	main()
	
