#!/usr/bin/env python
import numpy as np
import torch 
import torch.nn as nn
import wandb
import os
import math 
import random
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

from Dataset_multistep import Dataset_multiple_steps
from pytorch_model_summary import summary
from torch.utils.data import DataLoader,random_split
from tqdm import trange
from Drone_dyn import Drone
from PID_controller import PID

# Testrun Parameters
TIME_STEP = 0.01	# The sample time per time step [s]
SIM_TIME = 50		# Total length of simulation [s]
SETPOINT_UPDATE_STEP = 5
MIMIMAL_HEIGHT_CHANGE = 4
MASS = 1 			# Total mass of drone [kg]
g = -9.81 			# Gravitational constant [m/s^2]

# Testrun initialization
SETPOINT_Z = 0 	# Setpoint of height [m]
Z_INITIAL = 0. 		# Initial height [m]
DZ_INITIAL = 0 		# Initial velocity [m/s]

# Hyperparamaters
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64
PERCENT_TRAIN = 0.8
TRAINING_DATASET = "z_zref_kpe_kde_Thrust"
RNN_NEURONS = 40
RNN_LAYERS = 6
D_NETWORK = "gru"  #either rnn or gru
WEIGHT_INIT_SEED = 1

# Options for model run
SAVE_MODEL = True
WANDB_LOG = True
SHOW_MODEL = True
SHOW_PLOTS = False


class Simulation_NN():
	def __init__(self):
		torch.manual_seed(WEIGHT_INIT_SEED)
		self.network = NN_pid(RNN_NEURONS,RNN_LAYERS,D_NETWORK).to(device)
		self.run = wandb.init(project="my-test-project", mode=WANDB_MODE, reinit=True)
		set_config_wandb(self.network)
		self.setpoint_updated_number = -1 #Set to -1 to get a random setpoint immediately, otherwise 0 

	def NN_train(self):
		train_ds, validate_ds = initialize_NN(self.network)
		train_NN(self.network, train_ds, validate_ds)

	
	def NN_test(self):
		# Initialize overall MSE array
		self.MSE_arr = np.array([])
		global SETPOINT_Z

		for test in range(5):
			# Initialize testing variables
			self.states_nn = Drone(Z_INITIAL, DZ_INITIAL, MASS, TIME_STEP, g)
			self.rnn_hidden = torch.zeros(self.network.d1_num_layers, self.network.d1_hidden,dtype=torch.float32).to("cpu")
			self.network.to("cpu")
			self.timer = 0
			self.sim = True
			
			# Initialize for PID control output
			self.states_pid = Drone(Z_INITIAL, DZ_INITIAL, MASS, TIME_STEP, g)
			self.kp = 3
			self.ki = 0
			self.kd = 3
			self.pid = PID(self.kp,self.ki, self.kd)

			# Initialize arrays for plotting MSE
			self.thrst_p = np.array([])
			self.thrst_d = np.array([])
			self.h = np.array([])
			self.h_pid = np.array([])
			setpoint_arr = np.array([])

			# Start simulation
			while(self.sim):
				setpoint = SETPOINT_Z
				# Run through NN forward loop
				thrust_p, thrust_d, self.rnn_hidden = compute_NN(self.network,self.states_nn.get_z(), self.rnn_hidden, setpoint)
				thrust_total = thrust_p + thrust_d +(-g)*MASS
				self.states_nn.sim_dynamics(thrust_total)

				# Run through PID  loop
				thrust_pid = self.pid.compute(SETPOINT_Z,self.states_pid.get_z())
				self.states_pid.sim_dynamics(thrust_pid)

				# Append variable array that need to be plotted
				self.thrst_p = np.append(self.thrst_p,thrust_p)
				self.thrst_d = np.append(self.thrst_d,thrust_d)
				self.h = np.append(self.h,self.states_nn.get_z())
				self.h_pid = np.append(self.h_pid,self.states_pid.get_z())

				# Check if time limit is reached
				self.timer += 1
				if self.timer > SIM_TIME/TIME_STEP:
					print("SIM ENDED")
					self.sim = False
				
				# Update the setpoint if requested
				self.setpoint_updated_number, SETPOINT_Z = update_setpoint(SETPOINT_UPDATE_STEP, self.timer, TIME_STEP, self.setpoint_updated_number)
				setpoint_arr = np.append(setpoint_arr,SETPOINT_Z)

				
			# Plot height and thrust profile 
			self.time = np.arange(0,SIM_TIME+TIME_STEP,TIME_STEP)
			fig,(ax1,ax2) = plt.subplots(2,sharex=True)

			ax1.set_ylabel("Height")
			ax1.plot(self.time,setpoint_arr, color = 'r', linestyle = '-')
			ax1.plot(self.time,self.h,label="NN output")
			ax1.plot(self.time,self.h_pid,label="PID output")
			ax1.legend()

			ax2.set_ylabel("Thrust")
			ax2.plot(self.time,self.thrst_p+self.thrst_d+(-g*MASS), label="Total thrust")
			ax2.plot(self.time,self.thrst_p,label="Thrust P")
			ax2.plot(self.time,self.thrst_d, label="Thrust D")
			ax2.legend()
			if SHOW_PLOTS==True: plt.show()

			# Calculate the MSE between PID and NN response to target input
			MSE_h = np.mean(np.square(self.h-self.h_pid))
			self.MSE_arr = np.append(self.MSE_arr, MSE_h)
			print("MSE of height in plot = ", MSE_h)
			
			# Log the plot and the MSE per plot to wandb
			wandb.config.update({"MSE_"+str(test):MSE_h})
			wandb.log({"plot_height_step_"+str(test):fig})

		# Log overall MSE to wandb and finish current wandb run
		wandb.config.update({"MSE_total":np.mean(self.MSE_arr)})
		self.run.finish()
			

def update_setpoint(freq_update, timer, dt, prev_interval):
	global SETPOINT_Z
	new_setpoint = 0
	current_interval = math.floor(timer*dt/freq_update)
	
	if current_interval != prev_interval:
		while (new_setpoint == 0 or abs(new_setpoint-SETPOINT_Z) <= MIMIMAL_HEIGHT_CHANGE):
			new_setpoint = random.uniform(0,10)
		SETPOINT_Z = new_setpoint
	prev_interval = current_interval

	return prev_interval, SETPOINT_Z

class NN_pid(torch.nn.Module):
	def __init__(self,RNN_NEURONS,RNN_LAYERS,D_NETWORK):
		super(NN_pid, self).__init__()

		# Set the P NN parameters
		self.input_dim = 2
		self.error_dim = 1

		# Set the D NN paramaters
		self.d1_hidden = RNN_NEURONS
		self.d1_num_layers = RNN_LAYERS
		
		if D_NETWORK=="rnn":
			self.rnn_act_funct = "relu"
		else: self.rnn_act_funct = "None"


		# Initialize the P NN
		self.error = nn.Linear(self.input_dim,self.error_dim,bias=False)
		self.p1 = nn.Linear(self.error_dim,1,bias=False)

		# Initialize the D NN
		if D_NETWORK == "rnn":
			self.d1 = nn.RNN(self.error_dim,self.d1_hidden,self.d1_num_layers, batch_first =True, nonlinearity=self.rnn_act_funct, bias=False)
		if D_NETWORK == "gru":
			self.d1 = nn.GRU(self.error_dim,self.d1_hidden,self.d1_num_layers, batch_first =True, bias=False)
		self.d2 = nn.Linear(self.d1_hidden,1, bias=False)

	def forward(self, x, hidden):

		# Define the error layer
		out_error = self.error(x)

		# Define forward pass of P NN
		out_p = self.p1(out_error)

		# Define forward pass of D NN
		out_d,hidden = self.d1(out_error,hidden)
		out_d = self.d2(out_d)

		return out_p, out_d, hidden




def initialize_NN(model):
	# Initialize NN and optimizer
	model.optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
	model.train()
	wandb.watch(model, log="parameters", log_freq=10)

	# Initialize training ans test data set in a dictonary
	dataset = Dataset_multiple_steps("Sim_data/height_control_PID/"+ TRAINING_DATASET,5,[0,1],[2,3])
	train_data, val_data = random_split(dataset,[int(len(dataset)*PERCENT_TRAIN), len(dataset)- int(len(dataset)*PERCENT_TRAIN)], generator=torch.Generator().manual_seed(1))

	return train_data, val_data




def train_NN(model, train_data, val_data):

	#initialize total loss over epoch arrays
	loss_train_epochs = np.array([])
	loss_validate_epochs = np.array([])
	loss_train_p_epochs = np.array([])
	loss_train_d_epochs = np.array([])

	# Start the training and validation
	for ep in trange(EPOCHS, desc="Training NN"):

		# # # Prematurely stop training if early results do not look promissing
		if (ep==1 and (loss_train_p>32 or loss_train_d>25)) or (ep==11 and (loss_train_p>28 or loss_train_d>3)):
			wandb.config.update({"Epochs": ep},allow_val_change=True)
			print("abort, ealy results are bad")
			return

		# Initialize cummulative error for each epoch
		loss_train_cum = 0
		loss_train_cum_p = 0
		loss_train_cum_d = 0
		loss_val_cum = 0

		loss_criterion = nn.MSELoss(reduction='mean')

		# Initialize the dataloader class
		dl_train = DataLoader(train_data 
							, batch_size=BATCH_SIZE
							, drop_last=True
							, pin_memory=True
							, num_workers=3
							, shuffle=True)
		
		
		for input_train,label_train in dl_train:
			# t1 = time.perf_counter()
			# try: t8
			# except: print("t8 not defined yet")
			# else: print("training for loop data (all other times)= ", (t1-t8)*1000)

			input_train = input_train.to(device, non_blocking=True)
			label_train = label_train.to(device, non_blocking=True)

			# Initialize the grad and the hidden layer of the RNN
			model.optimizer.zero_grad(set_to_none=True)
			rnn_hidden = torch.zeros(model.d1_num_layers, BATCH_SIZE, model.d1_hidden, device=torch.device(device))

			# Forward training step (10 ms with wandb) d_out shape = [Batch_size, sequence length, 1]
			out_p ,out_d ,rnn_hidden = model(input_train,rnn_hidden)

			# Calculate the loss functions: Proportional and Derivative which is the MSE average of one full sequence length (20sec)
			# Unsquueze adds a new dimensions to the tensor [X,X].unsqueeze(2) -->[X,X,1]
			loss_train_p = loss_criterion(out_p, label_train[:,:,0].unsqueeze(2))
			loss_train_d = loss_criterion(out_d, label_train[:,:,1].unsqueeze(2))
			print("\nLoss_train_p = ",loss_train_p, "and Loss_train_d = ", loss_train_d)

			# Back propagation and update the weights (takes 10ms approx +10ms with wandb)
			loss_train_d.backward(retain_graph=True)

			# Set the .grad from the derivative path on the error weights to zero (since it is  "easier" to learn from proportional path)
			for name, param in model.named_parameters():
				if name == "error.weight":
					param.grad[0] = torch.tensor([[0.,0.]], device=device)
			loss_train_p.backward()
			
			# Update the weights using the optimizer 
			model.optimizer.step()

			# Calculate the cummulative loss over a epoch
			loss_train_p = loss_train_p.detach().to("cpu").numpy()
			loss_train_d = loss_train_d.detach().to("cpu").numpy()

			# print("After detach: Loss_train_p = ",loss_train_p, "and Loss_train_d = ", loss_train_d)
			loss_train_cum += loss_train_p + loss_train_d
			loss_train_cum_p += loss_train_p
			loss_train_cum_d += loss_train_d

			# t8 = time.perf_counter()

		# To get an average of the loss over 1 data sample (20s)
		iterations_dataloader_train = len(dl_train) 
		loss_train_cum = loss_train_cum/iterations_dataloader_train
		loss_train_cum_p = loss_train_cum_p/iterations_dataloader_train
		loss_train_cum_d = loss_train_cum_d/iterations_dataloader_train

		wandb.log({"loss_train_p": float(loss_train_cum_p),"loss_train_d": float(loss_train_cum_d)})
		# t11 = time.perf_counter()
		print("Epochs: ", ep," loss p cum = ", loss_train_cum_p," loss d cum = ", loss_train_cum_d)

		# Check if one of the losses contains a Nan, then exit training 
		if (math.isnan(loss_train_cum_d) or math.isnan(loss_train_cum_p)):
			print("A Nan occured --> close off this training session")
			return
		#--------------- Start valildation ------------------------------

		
		# Initialize the dataloader class
		dl_validate = DataLoader(val_data
								,batch_size = BATCH_SIZE
								,drop_last=True
								, pin_memory=True
								, num_workers=0)
		
		with torch.no_grad():
			for input_val,label_val in dl_validate:
				
				input_val = input_val.to(device, non_blocking=True)
				label_val = label_val.to(device, non_blocking=True)

				# Initialize the hidden layer of the RNN
				rnn_hidden = torch.zeros(model.d1_num_layers, BATCH_SIZE, model.d1_hidden,device=torch.device(device))

				# Forward step
				out_val_p, out_val_d, _ = model(input_val,rnn_hidden)

				# Calculate the loss functions: Proportional and Derivative
				loss_val_p = loss_criterion(out_val_p,label_val[:,:,0].unsqueeze(2))
				loss_val_d = loss_criterion(out_val_d,label_val[:,:,1].unsqueeze(2))

				# Add the cumulative loss of the validation datasets over a epoch
				loss_val_cum +=loss_val_p.to("cpu").detach().numpy() +loss_val_d.to("cpu").detach().numpy()

		# To get an average of the loss over 1 data sample (20s)
		iterations_dataloader_val = len(dl_validate)
		loss_val_cum = loss_val_cum/iterations_dataloader_val
	
		# Add the losses to an array to get the total cumulative error per epoch
		loss_train_epochs = np.append(loss_train_epochs,loss_train_cum)
		loss_train_p_epochs = np.append(loss_train_p_epochs,loss_train_cum_p)
		loss_train_d_epochs = np.append(loss_train_d_epochs,loss_train_cum_d)
		loss_validate_epochs = np.append(loss_validate_epochs,loss_val_cum)

		# print("Training loop = ", (t11-t0)*1000)
		# print("Validation loop = ", (t13-t11)*1000)
		# print("Appending to list = ", (t14-t11)*1000)

	model.eval()

	# Save the model of the final iteration to the "models" folder
	if (SAVE_MODEL ==True):
		torch.save(model.state_dict(),"models/"+ wandb.run.name + ".pyt")
	
	# Print the model structure of the final iteration to the terminal
	if (SHOW_MODEL==True):
		print(summary(model,torch.zeros(1,2,device=torch.device(device)),torch.zeros(model.d1_num_layers, model.d1_hidden,device=torch.device(device)),show_parent_layers=True, show_hierarchical=True))
		for name, param in model.named_parameters():
			print(name, "size is ", param.data.shape,  ".Values are: \n", param.data,"\n")
	
	# Plot the loss for the training and validation set 
	epochs_arr = np.arange(0,EPOCHS)
	fig = plt.figure()
	plt.title("Training and validation loss for all epochs")
	plt.xlabel("Epochs [-]")
	plt.ylabel("Loss [-]")
	plt.plot(epochs_arr,loss_train_epochs, label="Training data")
	plt.plot(epochs_arr,loss_validate_epochs, label= "Validation data")
	plt.plot(epochs_arr,loss_train_p_epochs, label= "Training P data")
	plt.plot(epochs_arr,loss_train_d_epochs, label= "Training D data")
	plt.yscale("log")
	plt.legend()
	if SHOW_PLOTS==True: plt.show()
	
	wandb.log({"Loss_plot_":fig})
	print("-----------Training finished ----------")



def compute_NN(model, z_measured,hidden,setpoint):
	
	# Convert all measured tensors to floats
	if (type(z_measured) != float):
		z_measured = z_measured.numpy()[0][0]

	# Set all input values to type tensor and dtype=float32
	input = np.array([[z_measured,setpoint]])
	input = input.astype(np.float32)
	input = torch.as_tensor(input)

	# Forward
	thrust_p,thrust_d, hidden_updated = model(input,hidden)
	
	# Detach such that tensors can be returned
	thrust_p = thrust_p.detach()
	thrust_d = thrust_d.detach()
	hidden_updated = hidden_updated.detach()

	return thrust_p, thrust_d, hidden_updated

def set_config_wandb(model):
	wandb.config.update({"lr": LEARNING_RATE, 
						"Epochs": EPOCHS,
						"Batch_size": BATCH_SIZE,
						"Percentage_train":PERCENT_TRAIN,
						"Training data": TRAINING_DATASET,
						"Input dim": model.input_dim, 
						"Error out": model.error_dim,
						"D1 RNN out": model.d1_hidden,
						"D1 RNN layer":model.d1_num_layers,
						"RNN act_funct":model.rnn_act_funct,
						"NN model": D_NETWORK,
						"Weight init seed": WEIGHT_INIT_SEED})


def main():
	sim = Simulation_NN()
	sim.NN_train()
	sim.NN_test()

# if __name__ == '__main__':
# 	# torch.multiprocessing.set_start_method('spawn')
# 	main()

# for LEARNING_RATE in [0.002]:
# 	for BATCH_SIZE in [6,10]:
# 		for RNN_NEURONS in [10,15]:
# 			for RNN_LAYERS in [1,2,4]:
# 				main()
# 				os.system('cls')

if __name__ == '__main__':

	if WANDB_LOG==True:
		WANDB_MODE = "online"	# "online", "offline" or "disabled"
	else:
		WANDB_MODE = "disabled"
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	print("Device = ", device)

	lr_list 		= [0.002,0.001,0.0005,0.0003]
	layer_list 		= [2,4,6,8]
	rnn_hidden_list = [4,10,20,40]
	batch_list 		= [8,16,32,64]

	for D_NETWORK in ["gru"]:
		for i in range(100):
				random.seed(a=None)
				seed = random.randint(0,100000)
				WEIGHT_INIT_SEED=seed
				LEARNING_RATE = random.choice(lr_list)
				RNN_LAYERS = random.choice(layer_list)
				RNN_NEURONS = random.choice(rnn_hidden_list)
				BATCH_SIZE = random.choice(batch_list)
				print("Total = ",i, "\n Batch size = ", BATCH_SIZE , "\n Rnn layers = ", RNN_LAYERS, "\n Rnn neurons = ", RNN_NEURONS, "\n Learning rate = ", LEARNING_RATE)
				main()
				os.system('cls')

# if __name__ == "__main__":
# 	if WANDB_LOG==True:
# 		WANDB_MODE = "online"	# "online", "offline" or "disabled"
# 	else:
# 		WANDB_MODE = "disabled"
# 	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 	print("Device = ", device)
# 	for _ in range(200):
# 		random.seed(a=None)
# 		seed = random.randint(0,100000)
# 		WEIGHT_INIT_SEED=seed
# 		print("Weigh initialization seed = ", WEIGHT_INIT_SEED)
# 		main()
# 		os.system('cls')