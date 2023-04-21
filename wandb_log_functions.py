import wandb
import numpy as np
import torch

def create_wandb_summary_table(network,run,spike_train,config):
	### print total spike count per neuron
	spike_count = []
	for neuron in range(config["NEURONS"]):
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


	summary_table = wandb.Table(rows=np.arange(0,config["NEURONS"]).tolist(), columns=names,data=data)
	run.log({"Trained parameters overview": summary_table})


def set_config_wandb(network, config):
	if config["LABEL_COLUMN_DATAFILE"][0] ==1:
		TYPE_NN = "Proportional"
	if config["LABEL_COLUMN_DATAFILE"][0] == 2:
		TYPE_NN = "Derivative"

	# Send the constants/hyper parameters to WANDB
	wandb.config.update({"lr_l1_neuron": config["LR_L1_NEURON"],
						"lr_l1_weights": config["LR_L1_WEIGHTS"],
						"lr_l2_neuron": config["LR_L2_NEURON"],
						"lr_l2_weights": config["LR_L2_WEIGHTS"], 
						"Epochs": config["EPOCHS"],
						"Batch size": config["BATCH_SIZE"],
						"Percentage train":config["PERCENT_TRAIN"],
						"Training data": config["DATASET_DIR"],
						"Neurons": config["NEURONS"], 
						"NN model": TYPE_NN,
						"Weight init seed": config["RANDOM_INIT_SEED"],
						"half weights negative":config["HALF_NEGATIVE_WEIGHTS"]})

def difference_learning_params(before, after):
	print("\n Difference in network parameters \n")
	for name in after:
		print(name)
		print((after[name]-before[name])) 

	print("\n \n")

def print_network_training_parameters(config):
	print("Parameter Bounds")
	print(config["PARAMETER_BOUNDS"])

	if config["INIT_SETTING"] == "random":
		print("\n Inital RANDOM values for the parameters")
		print(config["INITIAL_PARAMS_RANDOM"])	


def create_wandb_summary_table_EA(run,spike_train,config,final_parameters):
	### print total spike count per neuron
	spike_count = []
	for neuron in range(config["NEURONS"]):
		w1 = torch.flatten(final_parameters["l1.ff.weight"]).detach().numpy()[neuron]
		w2 = torch.flatten(final_parameters["l2.ff.weight"]).detach().numpy()[neuron]
		print("Spikecount neuron", neuron, " = ", np.sum(spike_train[:,neuron]), "\t w1 = ", np.round(w1,3), "\t & w2 = ", np.round(w2,3))
		spike_count.append(float(np.sum(spike_train[:,neuron])))

	data = []
	names = []
	for name, param in final_parameters.items():
		names.append(name)
		# ravel makes to converts the multi dim ndarrays to a single dimension
		params = param.detach().numpy().ravel()

		if len(params)==1:
			params= np.repeat(params,config["NEURONS"])
		data.append(list(params))
	names.append("Test Spike Count")
	data.append(spike_count)
	data = np.array(data).T.tolist()


	summary_table = wandb.Table(rows=np.arange(0,config["NEURONS"]).tolist(), columns=names,data=data)
	run.log({"Trained parameters overview": summary_table})