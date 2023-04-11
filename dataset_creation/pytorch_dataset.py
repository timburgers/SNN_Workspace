from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import torch
from os import listdir
import math
import numpy as np


class Dataset_derivative(Dataset):
	# This loads the data and converts it
	def __init__(self,config):
		self.root_dir = config["DATASET_DIR"]
		self.num_batches_per_data = config["NUM_BATCHES_PER_DATAFILE"]
		self.input_columns = config["INPUT_COLUMN_DATAFILE"]
		self.label_columns = config["LABEL_COLUMN_DATAFILE"]
		self.neurons = config["NEURONS"]

		self.file_names = [f for f in listdir(self.root_dir) if f[-4:] == ".csv"]
		# Get the number of samples in each data file
		samples_in_data = len(pd.read_csv(self.root_dir+"/"+self.file_names[0]))
		self.samples_per_batch = samples_in_data//self.num_batches_per_data
		
		self.input = torch.zeros(len(self.file_names)*self.num_batches_per_data,2)

		# Make a tensor [filename, starttime]
		for i in range(len(self.file_names)*self.num_batches_per_data):
			data_set_idx = math.floor(i/self.num_batches_per_data)
			start_time = i%self.num_batches_per_data*self.samples_per_batch
			self.input[i][0] = data_set_idx
			self.input[i][1] = start_time

		print("\n Dataset loaded = "+ self.root_dir )
		print("Data files found = "+ str(len(self.file_names)))
		print("Samples per data file = "+ str(samples_in_data))
		print("Sequence length = " + str(self.samples_per_batch))
		print("Total batches = ", len(self))
	# This returns the total amount of samples in your Dataset
	def __len__(self):
		return len(self.input[:,0])
	
	# This returns given an index the i-th sample and label
	def __getitem__(self, idx):
		batch_idx = self.input[idx]
		file_name = self.file_names[int(batch_idx[0].numpy())]
		start_time = int(batch_idx[1].numpy())

		# The first and second column contain the h and h_ref
		df_dataset = pd.read_csv(self.root_dir + "/" + file_name, usecols= self.input_columns,skiprows=start_time, nrows=self.samples_per_batch ,  header=0)
		dataset = df_dataset.to_numpy()                     #shape (seq_len,1)
		dataset = torch.tensor(dataset).float()

		# The thrid and fourth column contain the Kpe and Kde
		df_labels = pd.read_csv(self.root_dir + "/" + file_name, usecols= self.label_columns,skiprows=start_time, nrows=self.samples_per_batch , header=0)
		labels = torch.tensor(df_labels.to_numpy()).float()
		
		return dataset,labels




# neurons = 5
# number_batches_per_file = 1000
# percent_train = 0.8
# dataset = Dataset_derivative("Sim_data/derivative/dt0.002", number_batches_per_file, neurons)


# # print("Total length of the data set = ", len(dataset))
# train_data, val_data = random_split(dataset,[int(len(dataset)*percent_train), len(dataset)- int(len(dataset)*percent_train)], generator=torch.Generator().manual_seed(1))

# print(len(train_data), len(val_data))


#dataset[idx] = tuple(tensor(seq_length,1), tensor(seq_length,1)) --> tuple is (input, label)

# print(train_data[0])


# data_loader = DataLoader(dataset
# 						, batch_size=4
# 						, drop_last=True)

# for input,label in data_loader:
#     print("Input shape = ", input.shape, " And label shape = ",label.shape)
# # print(input)