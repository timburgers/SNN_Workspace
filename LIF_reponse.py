from SNN_LIF_LI_init import LIF_SNN
import torch
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import torch.nn as nn

neurons = 1
sim_time = 40
dataset_number =None # None is the self made 13s dataset
TIME_STEP = 0.01
stepwise_response = True

layer_setting = dict()
layer_setting["l1"] = {
    "recurrent":              False,
    "adaptive":               False,
    "clamp_v":                True ,
    "shared_leak_i":          False,
    "bias":                   True ,
    "shared_weight_and_bias": False
}
layer_setting["l2"]={
    "bias":                   False ,
    "shared_weight_and_bias": False}

thres = 2
w1=0.1
w2= 0.1
b1= 0.2
leak_i=0.9
leak_v1=0.5

leak_v2=0.9

def init_single(neurons,layer_set):
	init_param = {}

	init_param["l1_thres"]	= torch.ones(neurons).float()*thres
	init_param["l1_leak_v"]	= torch.ones(neurons).float()*leak_v1
	init_param["l2_leak"] 	= torch.ones(1).float()*leak_v2

	if layer_set["l1"]["recurrent"] == True:
		init_param["l1_weights_rec"]= torch.ones(neurons,neurons).float()

	if layer_set["l1"]["shared_weight_and_bias"] == True: 
		init_param["l1_weights"]= torch.ones((int(neurons/2),1)).float()*w1 	#NOTE: shape must be (neurons,1)
		init_param["l1_bias"]	= torch.ones((int(neurons/2))).float()*b1
	else:                                        
		init_param["l1_weights"]= torch.ones((neurons,1)).float()*w1			#NOTE: shape must be (neurons,1)
		init_param["l1_bias"]	= torch.ones(neurons).float()*b1
	
	if layer_set["l1"]["shared_leak_i"] == True:
		init_param["l1_leak_i"] = torch.ones((int(neurons/2)))*leak_i
	else: 
		init_param["l1_leak_i"]	= torch.ones(neurons).float()*leak_i 


	if layer_set["l2"]["shared_weight_and_bias"] == True:
		init_param["l2_weights"]= torch.ones((1,int(neurons/2))).float()*w2	#NOTE: shape must be (1, neurons)
	else:
		init_param["l2_weights"]= torch.ones((1,neurons)).float()*w2			#NOTE: shape must be (1, neurons)
	
	
	return init_param

### Step paramaters
range_step = [-1,1]
number_of_steps = 10
param_init = init_single(neurons,layer_setting)

controller = LIF_SNN(param_init,neurons,layer_setting)

def get_dataset(dataset_num, sim_time):
    # Either use one of the standard datasets
    if dataset_num != None: file = "/dataset_"+ str(dataset_num)
    else:                   file = "/test_dataset"
    
    input_data = pd.read_csv("Sim_data/height_control_PID/zref_norm_positive"+ file + ".csv", usecols=[0], header=None, skiprows=1, nrows=sim_time*(1/TIME_STEP))
    input_data = torch.tensor(input_data.values).float().unsqueeze(0) 	# convert from pandas df to torch tensor and floats + shape from (seq_len ,features) to (1, seq, feature)

    target_data = pd.read_csv("Sim_data/height_control_PID/zref_norm_positive" + file + ".csv", usecols=[0], header=None,  skiprows=1, nrows=sim_time*(1/TIME_STEP))
    target_data = torch.tensor(target_data.values).float()
    target_data = target_data[:,0]
    return input_data, target_data

def get_steps_data(sim_time,number_of_steps,range_step):
    input_data = np.array([])
    total_time = int(sim_time/TIME_STEP)

    for step_number in range(number_of_steps+1):
        for step_t in range(int(total_time/(number_of_steps+1))):
            input_data = np.append(input_data, range_step[0]+step_number*((range_step[1]-range_step[0])/number_of_steps))

    input_data = torch.from_numpy(input_data).to(torch.float32)
    input_data = input_data.unsqueeze(0)
    input_data = input_data.unsqueeze(2)
    return input_data


# Load the input data
if stepwise_response == True: input_data = get_steps_data(sim_time, number_of_steps, range_step)
else: input_data, _ = get_dataset(dataset_number,sim_time)



snn_states = torch.zeros(3, 1, controller.neurons) # frist dim in order [current,mempot,spike]
LI_state = torch.zeros(1,1)
l1_spikes, l1_state, control_output = controller(input_data, snn_states,LI_state)
actual_data = control_output[:,0,0].detach().numpy()


time_test = np.arange(0,np.size(actual_data)*TIME_STEP,TIME_STEP)
plt.plot(time_test, actual_data, color = "r", label="Output")
plt.plot(time_test,input_data[0,:,0], color = 'b',label="Input")
plt.title("Controller output and reference")
plt.grid()


    


### Calculate spiking sliding window count
spike_count_window = None
spikes_snn = l1_spikes[:,0,:] # spikes_izh of shape (timesteps,neurons)
window_size =100
sliding_window = nn.AvgPool1d(window_size,stride=1)

for neuron in range(spikes_snn.size(dim=1)):
    spike_count = spikes_snn[:,neuron].detach().numpy()
    spike_count = torch.tensor([spike_count])
    _slided_count = sliding_window(spike_count).detach().numpy()

    ### Fill in the slided window with zeros (at beginning and end to make it samen size)
    for _ in range(int(window_size/2)):
        _slided_count = np.insert(_slided_count,[0],[0])
    for _ in range(int(window_size/2)-1):
        _slided_count = np.append(_slided_count,[0])

    if not hasattr(spike_count_window, "shape"):
        spike_count_window = _slided_count
    else: spike_count_window = np.vstack((spike_count_window,_slided_count))
    plt.plot(time_test,_slided_count, "--", c="k",label = "Spike count neuron " + str(neuron))
plt.legend()
plt.show()

