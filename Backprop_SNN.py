import torch
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F 
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from Coding.Decoding import sliding_window, Linear_LI_filter
from wandb_log_functions import *




def train_SNN(network,dataset, device, config):
	network.train()

	### Intialize parameters from config
	epochs= config["EPOCHS"]
	percent_train= config["PERCENT_TRAIN"]
	learning_rate_default= config["LEARNING_RATE_DEFAULT"]
	batch_size= config["BATCH_SIZE"]
	random_seed= config["RANDOM_INIT_SEED"]
	bounds_of_parameters= config["PARAMETER_BOUNDS"]


	### Initialize optimizer with learning parameters
	optimizer_parameters = [{'params': network.l1.neuron.parameters(), 	'lr':config["LR_L1_NEURON"]},
							{'params': network.l2.neuron.parameters(),	'lr':config["LR_L2_NEURON"]},
							{'params': network.l1.ff.parameters(), 		'lr':config["LR_L1_WEIGHTS"]},
							{'params': network.l2.ff.parameters(), 		'lr':config["LR_L2_WEIGHTS"]}]
	
	optimizer = torch.optim.Adam(optimizer_parameters, lr = learning_rate_default)

	### Split dataset into training and validation data
	train_data, val_data = random_split(dataset,[int(len(dataset)*percent_train), len(dataset)- int(len(dataset)*percent_train)], generator=torch.Generator().manual_seed(random_seed))

	### Print training batches and iterations per epoch
	iter_per_epoch = int(len(train_data)/batch_size)
	print ("Total number of training batches = ", len(train_data), " & Iterations per epoch = ", iter_per_epoch, "\n")

	### Initialize total loss over epoch arrays
	loss_train_epochs = np.array([])


	### Start the training
	for ep in trange(epochs, desc="Training SNN"):
		loss_train_cum = 0
		loss_bounds_cum = 0

		dl_train = DataLoader(	train_data,
								batch_size = batch_size,
								drop_last=True,
								pin_memory=True,
								num_workers=0,
								shuffle=False)

		### Loop over batches: batchtrain --> (batch_size, seq_length, input neurons), batch label --> (batch_size, seq_length, output neurons)	
		for batch_train,label_train in dl_train: 
			batch_train = batch_train.to(device, non_blocking = True)
			label_train = label_train.to(device, non_blocking = True)

			### Convert shape of batch train from (batch_size, seq_len, 1) to (batch_size, seq_len, features)	
			# batch_train  = torch.Tensor.repeat(batch_train,(1,1,network.neurons))		

			#### Initialize neuron states (u, v, s) 
			snn_states = torch.zeros(3, batch_size, network.neurons, device=torch.device(device))
			snn_states[1,:,:] = -70			#initialize V
			snn_states[0,:,:] = -20 		#initialize U
			LI_state = torch.zeros(1, device=torch.device(device))

			### Initialize optimizer
			optimizer.zero_grad(set_to_none=True)


			#### Forward step (outputs in the shape (seq_len, batch_size, input features))
			spike_output,snn_states, decoded_output = network(batch_train,snn_states, LI_state) 

			### Convert shape of train data from (batch_size, seq_len ,feature) to (seq_len, batch_size, feature) since the underlaying neurons models output in that order
			label_train = torch.permute(label_train,(1,0,2)) 

			### Calculate the loss
			loss_train = F.mse_loss(decoded_output[100:,:,:], label_train[100:,:,:])
			# loss_train = F.smooth_l1_loss(decoded_output,label_train)
			print("loss of train data =",loss_train)

			loss_bounds = caluclate_loss_bounds(network, bounds_of_parameters)
			print("Bounds loss = ", loss_bounds)
			# loss_total = loss_train +loss_bounds


			### Back propagation (calulate the gradients) and update the weights/parameters
			loss_train.backward(retain_graph=True)

			print("\n Print Gradients = ")
			for name_, param_ in network.named_parameters():
				if param_.requires_grad:
					print (name_, param_.grad)
			print("\n")

			loss_bounds.backward()
			# torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10)

			# print("\n Print Gradients = ")
			# for name_, param_ in network.named_parameters():
			# 	if param_.requires_grad:
			# 		print (name_, param_.grad)
			# print("\n")

			optimizer.step()

			# list_learnable_parameters(network,True)
			### Calulate cumulative loss over an epoch
			loss_train = loss_train.to("cpu").detach().numpy()
			loss_bounds = loss_bounds.to("cpu").detach().numpy()
			loss_train_cum += loss_train
			loss_bounds_cum += loss_bounds
		

		### Check if a NAN is present in the network
		nan_check_in_parameters(network,ep, loss_train, loss_bounds)

		### Average the loss over the number of iterations of the dataloader and append to list
		loss_train_cum = loss_train_cum/len(dl_train)
		loss_bounds_cum = loss_bounds_cum/len(dl_train)
		loss_train_epochs = np.append(loss_train_epochs,loss_train_cum)

		wandb.log({"Training Loss": loss_train_cum})
		wandb.log({"Bounds Loss": loss_bounds_cum})
		wandb.log({"Leak_l2": network.l2.neuron.leak.to("cpu").detach().numpy()})
	
	return network, loss_train_epochs


# Function that checks is trainable parameters (weights or network parameters) is a Nan
def nan_check_in_parameters(network, current_epoch,loss_train,loss_bounds):
	for name, param in network.named_parameters():
		if param.requires_grad:
			if not torch.all(torch.isnan(param.data) == False):
				for name_, param_ in network.named_parameters():
					if param_.requires_grad:
						print (name_, param_.data)
				print("a NAN ocuurred in epoch ", current_epoch)
				print("The Nan occurred in ", name)
				print(torch.isnan(param.data))
				print("loss train = ", loss_train, " and bounds loss = ", loss_bounds)
				exit()




def test_SNN(network,train_loss,run,config):
	### Test network (on a single file)
	TIME_STEP_TEST = 0.002
	input_test = pd.read_csv(config["DATASET_DIR"]+ "/dataset_3.csv", usecols=config["INPUT_COLUMN_DATAFILE"], header=None, skiprows=[0])
	input_test = torch.tensor(input_test.values).float() 	# convert from pandas df to torch tensor and floats 
	# input_test = torch.tile(input_test,(1,NEURONS))		# convert from (seq_len,1) to (seq_len ,features)
	input_test = input_test.unsqueeze(0)					# shape from (seq_len ,features) to (1, seq, feature)										

	### Initialize the states(u, v, s)
	states_snn = torch.zeros(3,config["NEURONS"], device = torch.device("cpu"))
	state_LI = torch.zeros(1, device = torch.device("cpu"))

	### Call forward function
	network = network.to("cpu")
	spike_test, state_test, decoded_test = network(input_test,states_snn,state_LI)

	### Set target for the output decoded data, since input.detach.numpy is of shape (x,1) --> axis of diff is 0
	target_test = pd.read_csv(config["DATASET_DIR"] + "/dataset_3.csv", usecols=config["LABEL_COLUMN_DATAFILE"], header=None, skiprows=[0]).to_numpy()

	### Convert all torch.tensors to numpy element. snn shape = (seq_len, states, batch, features)
	spike = state_test[:,2,0,:].detach().numpy()
	sw_time, sw_spikes  = sliding_window(spike,TIME_STEP_TEST,0.01,TIME_STEP_TEST)

	input = input_test[0,:,0].detach().numpy()
	output = decoded_test[:,0,0].detach().numpy()
	time = np.arange(0,len(input)*TIME_STEP_TEST,TIME_STEP_TEST)
	

	create_wandb_summary_table(network,run,spike, config)

	fig,(ax1,ax2,ax3) = plt.subplots(3, sharex=True)
	fig.text(0.5, 0.04, 'Time (s)', ha='center')

	ax1.set_title("Input Current")
	ax1.plot(time,input)
	ax1.grid()

	ax2.set_title("Output spikes count (every 0.01 sec)")
	for input_neuron in range(np.size(sw_spikes,axis=1)):
		if input_neuron < int(config["NEURONS"]/2):
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
	if config["SHOW_PLOTS"] ==True: plt.show(block=False)
	
	fig2,ax2 = plt.subplots()
	epochs_list = np.arange(1,config["EPOCHS"]+1)
	ax2.plot(epochs_list,train_loss)
	ax2.set_title("Loss over epochs")
	ax2.set_xlabel("Epochs [-]")
	ax2.set_ylabel("Loss")
	ax2.grid()
	if config["SHOW_PLOTS"] ==True: plt.show()

	### Log the first plot to wandb
	run.log({"Test trial": fig})









def caluclate_loss_bounds(network, bounds_and_parameters):
	total_loss = 0
	key_list = list(bounds_and_parameters.keys())

	### Loop over all set bounds
	for idx in range(len(key_list)):
		### Initialize name to none (to check if variabel is set later)
		param_data = None

		### Loop over trained paramaters and find varaibles and min/max parameters
		for name, param_ in network.named_parameters():
			if name == key_list[idx]:
				param_data = param_
				min = bounds_and_parameters[name][0]
				max = bounds_and_parameters[name][1]
				break
		if param_data == None:
			exit(str(key_list[idx]) +" not found in the named parameters of the network")

		
		# Calculate the center of the interval and the distance from the center to the max/min (aka border)
		if None not in (min, max):
			center = (min+max)/2
			border = (max-min)/2

			# Center the parameters around zero (by "- middle") and mirror all negative values to the positive axis
			param_data = abs(param_data - center)
			
			# Set the loss for all values within the interval to zero and outside use a linear distribution
			loss = (torch.clamp(param_data, min=border) - border).sum()

		elif min != None:
			param_data = param_data - min
			loss = (torch.clamp(param_data, max = 0)*-1).sum()
			loss = loss*10
		
		elif max != None:
			param_data = param_data - max
			loss = torch.clamp(param_data, min = 0).sum()
			loss = loss*10

		### Check if both inputs are not none
		if min != None or max != None:
			total_loss += loss

	
	return total_loss