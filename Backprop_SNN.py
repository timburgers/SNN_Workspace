import torch
import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F 
from dataset_creation.pytorch_dataset import Dataset_derivative
import wandb
from SNN_Izhickevich import list_learnable_parameters




def train_SNN(network, dataset, epochs, percent_train, learning_rate, batch_size, random_seed, device):
	network.train()

	### Initialize optimizer
	optimizer = torch.optim.Adam(network.parameters(), lr = learning_rate)

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

		dl_train = DataLoader(	train_data,
								batch_size = batch_size,
								drop_last=True,
								pin_memory=True,
								num_workers=0,
								shuffle=True)

		### Loop over batches: batchtrain --> (batch_size, seq_length, input neurons), batch label --> (batch_size, seq_length, output neurons)	
		for batch_train,label_train in dl_train: 
			batch_train = batch_train.to(device, non_blocking = True)
			label_train = label_train.to(device, non_blocking = True)

			### Convert shape of batch train from (batch_size, seq_len, 1) to (batch_size, seq_len, features)	
			# batch_train  = torch.Tensor.repeat(batch_train,(1,1,network.neurons))		

			#### Initialize neuron states
			snn_states = torch.zeros(3, batch_size, network.neurons, device=torch.device(device))
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
			# print("loss of train data =",loss_train)
			
			### Back propagation (calulate the gradients) and update the weights/parameters
			loss_train.backward()

			# print("\n Print Gradients = ")
			# for name_, param_ in network.named_parameters():
			# 	if param_.requires_grad:
			# 		print (name_, param_.grad)
			# print("\n")

			optimizer.step()

			list_learnable_parameters(network,True)
			### Calulate cumulative loss over an epoch
			loss_train = loss_train.to("cpu").detach().numpy()
			loss_train_cum += loss_train
		

		### Check if a NAN is present in the network
		nan_check_in_parameters(network,ep)

		### Average the loss over the number of iterations of the dataloader and append to list
		loss_train_cum = loss_train_cum/len(dl_train)
		loss_train_epochs = np.append(loss_train_epochs,loss_train_cum)
		wandb.log({"Training Loss": loss_train_cum})
		wandb.log({"Leak_l2": network.l2.neuron.leak.to("cpu").detach().numpy()})
	

	return network, loss_train_epochs



# Function that checks is trainable parameters (weights or network parameters) is a Nan
def nan_check_in_parameters(network, current_epoch):
	for name, param in network.named_parameters():
		if param.requires_grad:
			if not torch.all(torch.isnan(param.data) == False):
				for name_, param_ in network.named_parameters():
					if param_.requires_grad:
						print (name_, param_.data)
				print("a NAN ocuurred in epoch ", current_epoch)
				print("The Nan occurred in ", name)
				print(torch.isnan(param.data))
				exit()