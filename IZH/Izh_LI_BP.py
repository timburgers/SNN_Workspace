# Verified using Figure F (Spike frequency adaption)
# Input data: http://www.izhikevich.org/publications/figure1.m
# Output data: http://www.izhikevich.org/publications/figure1.pdf

import torch
from dataset_creation.pytorch_dataset import Dataset_derivative
from SNN_Izh_LI_init import Izhikevich_SNN, list_learnable_parameters, initialize_parameters
from Backprop_SNN import train_SNN, test_SNN
from wandb_log_functions import *
import os
import wandb
import yaml


def main(config):

	### Initialize SNN + Dataset
	param_init = initialize_parameters(config)
	SNN_izhik = Izhikevich_SNN(param_init, device, config).to(device)
	dataset = Dataset_derivative(config)  

	### Set weight & biases
	run = wandb.init(project= "SNN_Izhikevich_P", mode=WANDB_MODE, reinit=True, config=config)	
	set_config_wandb(SNN_izhik, config)
	wandb.watch(SNN_izhik, log="parameters", log_freq=1)

	### Train the network
	# param_before_training = list_learnable_parameters(SNN_izhik, show_results=True)
	param_before_training = list_learnable_parameters(SNN_izhik, show_results=False)
	trained_network, train_loss = train_SNN(SNN_izhik, dataset, device, config)
	param_after_training = list_learnable_parameters(trained_network, show_results=False)
	
	### Calculate the difference between start and final learnable parameters
	difference_learning_params(param_before_training,param_after_training)
	param_after_training = list_learnable_parameters(trained_network, show_results=True)

	### Test the network
	test_SNN(trained_network, train_loss,run, config)

	print_network_training_parameters(config)



if __name__ == '__main__':
	os.system('cls')

	### Read config file
	with open("config_Izh_LI_BP.yaml","r") as f:
		config = yaml.safe_load(f)

	if config["WANDB_LOG"]==True:
		WANDB_MODE = "online"	# "online", "offline" or "disabled"
	else:
		WANDB_MODE = "disabled"
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	device = "cpu"
	print("Device = ", device)

	main(config)
	
