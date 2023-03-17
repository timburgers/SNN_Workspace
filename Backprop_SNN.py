from SNN_Izhickevich import Izhikevich_SNN
import torch
import numpy as np
from tqdm import trange
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F 
import pandas as pd
import matplotlib.pyplot as plt
from Coding.Decoding import sliding_window
import torch.nn as nn


class Dataset_derivative(Dataset):
	# This loads the data and converts it
	def __init__(self,input_signal, num_input, num_output):
		df_input = (pd.read_csv(input_signal,usecols=[1], header=0)).to_numpy()
		df_labels = (pd.read_csv(input_signal,usecols=[2], header=0)).to_numpy()

		# Convert input/output into multiple collomn vector (as many as input/output neurons there are)
		df_input = np.tile(df_input,(1,num_input))
		df_labels = np.tile(df_labels,(1,num_output))
		
			
		self.dataset = torch.tensor(df_input).float()
		self.labels = torch.tensor(df_labels).reshape(-1,1).float()
	
	# This returns the total amount of samples in your Dataset
	def __len__(self):
		return len(self.dataset)
	
	# This returns given an index the i-th sample and label
	def __getitem__(self, idx):
		return self.dataset[idx],self.labels[idx]



def train_SNN(network, data_sets, epochs, batch_size, perc_train):

	# Initialize NN and optimizer
	optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
	network.train()
	data_files = 100

	# Initialize training and test data set in a dictonary
	d = {}
	train_ds    = []
	test_ds     = []
	val_ds = []

	# split the NOT training data, into test and validate datasets
	perc_val = (1-perc_train)/2
	perc_test = (1-perc_train)/2

	for x in range(0,data_files):
		d["ds{0}".format(x)] = Dataset_derivative(data_sets + "/test_derivative_sin_" + str(x) +".csv",2,2)

		# Split the dataset in seperate: training, testing and validating sets
		if x < perc_train*data_files:
			train_ds.append(d["ds{0}".format(x)])
		elif x < (perc_test+perc_train)*data_files:
			test_ds.append(d["ds{0}".format(x)])
		else:
			val_ds.append(d["ds{0}".format(x)])

	print ("Training data set contains ", len(train_ds), " files")
	print ("Test data set contains ", len(test_ds), " files")
	print ("Validation data set contains ", len(val_ds), " files")
	

	SNN_layers = 1


	# Start the training
	for ep in trange(epochs, desc="Training SNN"):
		loss_train_cum = 0
		for ds in train_ds :
			# for data in ds:
			#     print("Dataset size = ", data)
			dl_train = DataLoader(ds,batch_size=batch_size)

			snn_states = torch.zeros(3, network.neurons)

			# for data in dl_train:
			# 	print(data[0].shape)
			# 	break
			for batch_train,label_train in dl_train:					
				# print("Batch size = ", batch_train.size())
				# print("Label size = ", label_train.size())

				snn_states = torch.zeros(3, network.neurons)

				# Forward step
				optimizer.zero_grad()
				outp,snn_states = network(batch_train,snn_states)

				

				# transpose ouptut from (samples, neurons) --> (neurons, samples)
				outp = torch.transpose(outp,0,1)

				window_size =100
				stride = 1 
				decoding_method = nn.AvgPool1d(window_size, stride)
				decoded_output = decoding_method(outp)
				decoded_output = torch.transpose(decoded_output,0,1)
				


				label_train = label_train[int(window_size/2):int(-window_size/2),0]
				# Calculate the loss
				loss_train = F.mse_loss(decoded_output, label_train)
				# print("loss of train data =",loss_train)
				
				# Back propagation and update the weights
				loss_train.backward()

				# for p in model.parameters():
				# 	print("loss = ",p.grad.norm())
				optimizer.step()

				loss_train_cum += loss_train.detach().numpy()
			#print("loss of train data cum =",loss_train_cum)

	return network