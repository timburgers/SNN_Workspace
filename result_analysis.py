import torch
import pygad
import pygad.torchga as torchga
from IZH.SNN_Izh_LI_init import Izhikevich_SNN
from SNN_LIF_LI_init import L1_Decoding_SNN, Encoding_L1_Decoding_SNN
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.fft import fft, fftfreq
import os
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore")
import math
from torchmetrics import PearsonCorrCoef
import copy
from sim_dynamics.Dynamics import Blimp
from LIF_EVOTORCH import get_dataset, run_controller, run_controller_dynamics, evaluate_fitness
import copy

# for dataset_number in range(10):
sim_time = 300
dataset_number = None                                                  # None is the test_dataset
filename = 271                                                       #None --> highest number, or int or str (withou .pkl)
folder_of_model = "Blimp"                                               # all folder under the folder Results_EA
lib_algorithm = "evotorch"                                              # evotorch or pygad
SNN_TYPE = "LIF"                                                        # either LIF or IZH
window_size =6      #NOTE: even numbers
#config["DATASET_DIR"] = "Sim_data/height_control_PID/fast_steps"


####################
exclude_non_spiking_neurons = False
# excluded_neurons=[1,2,23,24,29,30]
excluded_neurons =[]
new_dataset = None
new_dataset_number = 0
new_input_column = []
new_target_column = []

create_plots                    = False
create_table                    = False
plot_with_best_testrun          = True  #True: solution = best performance on manual dataset      False: solution = best performance overall (can be easy dataset)
muliple_test_runs_error_plot    = False  
plot_last_generation            = False

colored_background              = True
spike_count_plot                = True





#Pick last file name
if filename == None:
    all_files = os.listdir("Results_EA/"+folder_of_model)
    splitted_files = [int(f.split("-")[0]) for f in all_files]
    max_ind = splitted_files.index(max(splitted_files))
    filename = all_files[max_ind].split(".")[0] #get rid of .pkl
    print("\nFilename used = ", filename)

if type(filename) == int:
    all_files = os.listdir("Results_EA/"+folder_of_model)
    splitted_files = [int(f.split("-")[0]) for f in all_files]
    index_file = splitted_files.index(filename)
    filename = all_files[index_file].split(".")[0] #get rid of .pkl
    print("\nFilename used = ", filename)


# Run the simulation
def run_sim(fitness_mode, config, controller,solution,input_dataset, save=True):
    #The solution is defined in a numpy array shape
    sys_output = None

    final_parameters =torchga.model_weights_as_dict(controller, solution)
    controller.load_state_dict(final_parameters)

    if encoding_layer: controller.l0.init_reshape()
    controller.l1.init_reshape()
    controller.l2.init_reshape()

    if  fitness_mode== 1: #Only controller simlation
        state_l2_arr, state_l1_arr, state_l0_arr = run_controller(controller,input_dataset, save_mode=save)
        fitness_measured = state_l2_arr
        error_arr = torch.flatten(input_dataset).detach().numpy()

    elif fitness_mode == 2 or fitness_mode == 3:#Also simulate the dyanmics
        fitness_measured, error_arr, state_l0_arr, state_l1_arr, state_l2_arr = run_controller_dynamics(config,controller,input_dataset, save_mode=save)

    return fitness_measured, error_arr, state_l0_arr ,state_l1_arr ,state_l2_arr, final_parameters

def torch_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

# Function that assigns the correct varaible to eachother, based on wheter they are adaptive or not
def set_variables(adaptive,control_state):
    # Convert to tensors since it is required for the other part of the script
    if adaptive: 
        current = control_state[:,0,0,:]  #(timesteps,neurons)
        mem_pot = control_state[:,1,0,:]  #(timesteps,neurons)
        thres   = control_state[:,2,0,:]  #(timesteps,neurons)
        spikes  = control_state[:,3,0,:]  #(timesteps,neurons)

    else:                                         
        current = control_state[:,0,0,:]  #(timesteps,neurons)
        mem_pot = control_state[:,1,0,:]  #(timesteps,neurons)
        spikes  = control_state[:,2,0,:]  #(timesteps,neurons)
        thres = None
    return current,mem_pot,thres,spikes


#########################################################################################
if lib_algorithm == "pygad":
    ### load the ga_instance of the last generation
    loaded_ga_instance = pygad.load("Results_EA/"+ folder_of_model +"/" + filename)
    loaded_ga_instance.parallel_processing = None
    solution = loaded_ga_instance.best_solutions[-1]

if lib_algorithm == "evotorch":
    pickle_in = open("Results_EA/"+ folder_of_model +"/" + filename+".pkl","rb")
    dict_solutions = pickle.load(pickle_in)
    solution = dict_solutions["best_solution"]
    solutions_error = dict_solutions["error"]
    step_size = dict_solutions["step_size"]

    if plot_with_best_testrun == True:
        solutions = dict_solutions["test_solutions"]
        
        
        generations = dict_solutions["generations"]

        best_sol_ind = np.argmin(solutions_error)
        solution = solutions[best_sol_ind+1] #+1 since first of solution_testrun is only zeros
        if plot_last_generation == True:
            solution = solutions[-1] #+1 since first of solution_testrun is only zeros
            print("Plot shown with last generation")

        # Check if the best solution is the last generation
        if best_sol_ind == len(solutions_error)-1:
            best_gen = generations
        else: best_gen = (best_sol_ind)*step_size
        if plot_last_generation == False: print("Best solutions of the intermidiate testrun implementation is found at ", best_gen, " generations")
    else: print("Solution: Best evaluated solution (note: can be the result of an easy training dataset)")


### Select LIF or IZH mode
if SNN_TYPE == "LIF":
    config = dict_solutions["config"]
    number_of_neurons = config["NEURONS"]
    #//TODO GET rid of beun fixes here
    config["LAYER_SETTING"]["l0"]["shared_leak_iv"] = False
    config["LAYER_SETTING"]["l1"]["shared_leak_iv"] = False
    config["LAYER_SETTING"]["l0"]["shared_thres"] = False
    encoding_layer = config["LAYER_SETTING"]["l0"]["enabled"]
    if encoding_layer: controller = Encoding_L1_Decoding_SNN(None, config["NEURONS"], config["LAYER_SETTING"])
    else:              controller = L1_Decoding_SNN(None, config["NEURONS"], config["LAYER_SETTING"])

    #//TODO GET rid of beun fixes here
    controller.fitness_func = "mse"


elif SNN_TYPE == "IZH":
    with open("config_Izh_LI_EA.yaml","r") as f:
        config = yaml.safe_load(f)
    controller = Izhikevich_SNN(None, "cpu", config)

    ### Initialize neuron states (U, V, spikes) 
    snn_states = torch.zeros(3, 1, controller.neurons)
    snn_states[0,:,:] = -20 		#initialize U
    snn_states[1,:,:] = -70			#initialize V
    LI_state = torch.zeros(1,1)



time_step = config["TIME_STEP"]

total_time_steps = int(sim_time/time_step)

# Initialize varibales from problem Class
print("\nDataset used = ", config["DATASET_DIR"], "\nDatasetnumber = ", dataset_number)
input_data, fitness_target = get_dataset(config, dataset_number, sim_time)
fitness_mode = config["TARGET_FITNESS"]


##################           RUN SIM                ########################################################
fitness_measured, error_arr, state_l0_arr ,state_l1_arr ,state_l2_arr ,trained_parameters = run_sim(fitness_mode, config,controller,solution,input_data,True)
all_parameters = controller.state_dict()       #Convert the dict of only trained paramaters, to a dict with all parameters per neuron
l1_current,l1_mem_pot, l1_thres,l1_spikes = set_variables(config["LAYER_SETTING"]["l1"]["adaptive"],state_l1_arr)
l0_current,l0_mem_pot, _ ,l0_spikes = set_variables(False,state_l0_arr)
######################################################
if excluded_neurons or exclude_non_spiking_neurons:

    manually_excluded_neurons = 0
    if excluded_neurons: #if there are entrys
        excluded_neurons = [i - 1 for i in excluded_neurons] #since ind start at zero for further calulcations
        manually_excluded_neurons = len(excluded_neurons)

    # Check which neurons are non-spiking
    if exclude_non_spiking_neurons ==True:
        for neuron in range(controller.neurons):
            if not l1_spikes[:,neuron].any():
                excluded_neurons.append(neuron)
    print("\nNon firing neurons excluded = ", (len(excluded_neurons)-manually_excluded_neurons), "\nManually excluded neurons = ", manually_excluded_neurons)
    
    # Pre process the second run with the selected neurons only
    new_number_of_neurons = number_of_neurons-len(excluded_neurons)
    sparse_np_solution = np.array([])
    for name, param in all_parameters.items():
        sparse_param = param                                        #for the parameters that do no pass if statement
        # //TODO Does not work for recurrent, diagonal input matrix
        if torch.flatten(param).shape[0] == number_of_neurons:
            sparse_param=torch_delete(param,excluded_neurons)
        sparse_np_solution = np.append(sparse_np_solution,sparse_param)
  

    sparse_config = copy.deepcopy(config)
    sparse_config["NEURONS"] = new_number_of_neurons
    sparse_config["LAYER_SETTING"]["l1"]["shared_leak_iv"]          = False          #Set all shared weight to false, since then no parameters are shared in the second simulation
    sparse_config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"] = False
    sparse_config["LAYER_SETTING"]["l2"]["shared_weight_and_bias"] = False
    if new_dataset is not None:
        sparse_config["DATASET_DIR"] = "Sim_data/height_control_PID/" + new_dataset
        dataset_number = new_dataset_number
    if new_input_column or new_target_column:
        sparse_config["ALTERNATIVE_INPUT_COLUMN"] = new_input_column
        sparse_config["ALTERNATIVE_TARGET_COLUMN"] = new_target_column

    sparse_controller = L1_Decoding_SNN(None,new_number_of_neurons, sparse_config["LAYER_SETTING"])
    print("\nNEW Dataset used = ", sparse_config["DATASET_DIR"], "\nNEW Datasetnumber = ", dataset_number)
    new_input_data, fitness_target = get_dataset(sparse_config, dataset_number, sim_time)

    fitness_measured, error_arr, state_l0_arr ,state_l1_arr ,state_l2_arr ,all_sparse_parameters = run_sim(fitness_mode, sparse_config,sparse_controller,sparse_np_solution,new_input_data, True)
    l1_current,l1_mem_pot, l1_thres,l1_spikes = set_variables(config["LAYER_SETTING"]["l1"]["adaptive"], state_l1_arr)
    l0_current,l0_mem_pot, _ ,l0_spikes = set_variables(False,state_l0_arr)

    controller = sparse_controller
    number_of_neurons = new_number_of_neurons
    input_data = new_input_data
    all_parameters = all_sparse_parameters

    sparse_config["TARGET_FITNESS"] = 1 #since the target of mode 1 is the pid response
    _, ideal_pid_response = get_dataset(sparse_config, dataset_number, sim_time)

#####################################################
else:
    config["TARGET_FITNESS"] = 1 #since the target of mode 1 is the pid response
    _, ideal_pid_response = get_dataset(config, dataset_number, sim_time)



# Calculate the fitness value
fitness_value = evaluate_fitness(controller.fitness_func, fitness_measured, fitness_target)
print("Fitness value = ", np.round(fitness_value.item(),5))

# # calculate the splitted 
# mse = torch.nn.MSELoss()
# pearson = PearsonCorrCoef()
# #Evaluate fitness using MSE and additionally pearson if there should be a linear correlation between target and output
# _fitness_target = torch.flatten(fitness_target)
# _fitness_measured = torch.from_numpy(fitness_measured)
# mse = mse(_fitness_measured,_fitness_target)
# print("MSE = ", np.round(mse.item(),5))
# if fitness_mode == 1 or fitness_mode == 3:
#     pearson =  (1-pearson(_fitness_measured,_fitness_target)) #pearson of 1 means linear correlation
#     print("Pearson = ", np.round(pearson.item(),5))



# # Print the parameters of the best solution to the terminal
# for key, value in final_parameters.items():
#     print(key, value)

if fitness_mode == 1:   label_fitness_measured = "SNN output"; label_fitness_target = "PID output"
if fitness_mode == 2:   label_fitness_measured = "Blimp Height SNN"; label_fitness_target = "Blimp Height Reference"
if fitness_mode == 3:   label_fitness_measured = "Blimp Height SNN"; label_fitness_target = "Blimp height PID"

title = "Controller Response"
time_test = np.arange(0,sim_time,config["TIME_STEP"])
if fitness_mode == 2 or fitness_mode == 3:
    title = "Height control of the Blimp"
if fitness_mode == 3:
    plt.plot(time_test, torch.flatten(input_data), linestyle = "--", color = "r", label = "Reference input")

plt.plot(time_test, fitness_measured, color = "b", label=label_fitness_measured)
plt.plot(time_test, fitness_target, color = 'r',label=label_fitness_target)
plt.plot(time_test, error_arr, label = "Error" )
plt.title(title)
plt.grid()
plt.legend()
if create_plots == False and create_table == False and muliple_test_runs_error_plot == False :
    plt.show()



def calc_spike_moving_averge(spike_array):
    ### Calculate spiking sliding window count
    spike_count_window = None

    sliding_window = nn.AvgPool1d(window_size,stride=1)
    spike_array_torch = torch.from_numpy(spike_array).unsqueeze(0) #(batch, sequence L, neurons)

    for neuron in range(spike_array_torch.size(dim=2)):
        _slided_count = sliding_window(spike_array_torch[:,:,neuron]).detach().numpy()

        ### Fill in the slided window with zeros (at beginning and end to make it samen size)
        for _ in range(int(window_size/2)):
            _slided_count = np.insert(_slided_count,[0],[0])
        for _ in range(int(window_size/2)-1):
            _slided_count = np.append(_slided_count,[0])

        if not hasattr(spike_count_window, "shape"):
            spike_count_window = _slided_count
        else: spike_count_window = np.vstack((spike_count_window,_slided_count))
    return spike_count_window


if create_plots == True:
    # Create function that can get the indices where a plot is either nega/positive (mode= nonezero) or zero (mode = zero)
    def get_idx_of_non_zero(array,mode):
        if mode == "nonzero":
            filtered_array = np.where(array!=0)[0]
        if mode == "zero":
            filtered_array = np.where(array==0)[0]

        filt_start_idx = np.array([])
        filt_end_idx = np.array([])

        for idx in range(len(filtered_array)):
            if idx == 0:
                filt_start_idx = np.append(filt_start_idx,filtered_array[idx])
            elif filtered_array[idx]-prev_value != 1:
                filt_start_idx = np.append(filt_start_idx,filtered_array[idx])
                filt_end_idx = np.append(filt_end_idx,filtered_array[idx-1])
            if idx ==len(filtered_array)-1:
                filt_end_idx = np.append(filt_end_idx,filtered_array[idx])
            prev_value = filtered_array[idx]

        return filt_start_idx, filt_end_idx

    #Get list for the colored background
    input_neg = np.clip(error_arr,a_min=None, a_max=0)
    input_pos = np.clip(error_arr, a_min=0, a_max=None)

    pos_idx_start, pos_idx_end = get_idx_of_non_zero(input_pos, mode="nonzero")
    neg_idx_start, neg_idx_end = get_idx_of_non_zero(input_neg, mode="nonzero")
    zero_idx_start, zero_idx_end = get_idx_of_non_zero(input_data, mode="zero")

    if config["LAYER_SETTING"]["l0"]["enabled"] == False:
        ### Create the sperated spiking plots
        number_of_plots = math.ceil(number_of_neurons/10)
        for idx_plot in range(number_of_plots):
            if idx_plot == number_of_plots-1:
                neurons_in_plot = number_of_neurons - idx_plot*10
            else: neurons_in_plot = 10 
                
            # Start creating the Figure
            time_arr = np.arange(0,sim_time,time_step)
            axis1 = plt.figure(layout="constrained").subplot_mosaic(
                [
                    ["0,0","0,1","0,2","0,3"],
                    ["1,0","1,1","1,2","1,3"],
                    ["2,0","2,1","2,2","2,3"],
                    ["3,0","3,1","3,2","3,3"],
                    ["4,0","4,1","4,2","4,3"],
                    ["input","input","input","input"]
                ],
                sharex=True
            )

            ### plot the raster of U and V of the 10 neurons (5xV, 5xU, 5xV, 5xU)
            for column in range(2):
                neuron = 0+10*idx_plot
                for row in range(neurons_in_plot):
                    
                    if column ==0 or column ==2:
                        y = l1_mem_pot[:,neuron]

                    # Plot in the second column
                    if column ==1 or column ==3:
                        if spike_count_plot== True:
                            y=spike_count_window[neuron,:]

                        else: 
                            y = l1_current[:,neuron] #LIF --> current and for IZH --> recovery variable
                    
                    if row==5:
                        column = column + 2
                    if row>4:
                        row = row -5
                    
                    axis1[str(row)+","+str(column)].plot(time_arr,y)

                    axis1[str(row)+","+str(column)].xaxis.grid()

                    ### only plot in V plots
                    if column ==0 or column ==2:
                        # Show the threshold 
                        if config["LAYER_SETTING"]["l1"]["adaptive"]:
                            t = l1_thres[:,neuron]
                            base_t = all_parameters["l1.neuron.base_t"].detach().numpy()[neuron]
                            add_t = all_parameters["l1.neuron.add_t"].detach().numpy()[neuron]
                            threshold = t *add_t + base_t
                            axis1[str(row)+","+str(column)].plot(time_arr, threshold, color="r")
                        else:
                            threshold = all_parameters["l1.neuron.thresh"].detach().numpy()[neuron]
                            axis1[str(row)+","+str(column)].axhline(threshold,color="r")

                        # Plot the different background, corresponding with target sign
                        if colored_background == True:
                            for i in range(len(pos_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(pos_idx_start[i]*time_step, pos_idx_end[i]*time_step, facecolor="g", alpha= 0.2)
                            for i in range(len(neg_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(neg_idx_start[i]*time_step, neg_idx_end[i]*time_step, facecolor="r", alpha= 0.2)
                            for i in range(len(zero_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(zero_idx_start[i]*time_step, zero_idx_end[i]*time_step, facecolor="k", alpha= 0.2)
                    
                    ### only plot in U plots
                    if column ==1 or column ==3:
                        axis1[str(row)+","+str(column)].axhline(0,linestyle="--",color="k")
                        
                        # Plot the different background, corresponding with target sign
                        if colored_background == True and spike_count_plot==True:
                            for i in range(len(pos_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(pos_idx_start[i]*time_step, pos_idx_end[i]*time_step, facecolor="g", alpha= 0.2)
                            for i in range(len(neg_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(neg_idx_start[i]*time_step, neg_idx_end[i]*time_step, facecolor="r", alpha= 0.2)
                            for i in range(len(zero_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(zero_idx_start[i]*time_step, zero_idx_end[i]*time_step, facecolor="k", alpha= 0.2)
                    neuron = neuron +1
                column = 0

            time_arr = np.arange(0,sim_time,time_step)
            ### Plot the lowest figure
            axis1["input"].plot(time_arr,error_arr, label = "SNN input")
            axis1["input"].plot(time_arr,state_l2_arr, label = "SNN output")
            axis1["input"].plot(time_arr,ideal_pid_response, label = "PID reponse")
            axis1["input"].axhline(0,linestyle="--", color="k")
            axis1["input"].xaxis.grid()
        plt.legend()

    if config["LAYER_SETTING"]["l0"]["enabled"] == True:

        neurons_l0 = config["LAYER_SETTING"]["l0"]["neurons"]
        neurons_l1 = config["NEURONS"]
        neurons_l0 = config["NEURONS"]        #//TODO remove this line!!!

        l0_spike_count = calc_spike_moving_averge(l0_spikes)
        l1_spike_count = calc_spike_moving_averge(l1_spikes)

        ### Create the sperated spiking plots
        number_of_plots = max(math.ceil(neurons_l0/5), math.ceil(neurons_l1/5))
        for idx_plot in range(number_of_plots):
            l0_neurons_in_plot = min(5, max(neurons_l0-idx_plot*5,0))
            l1_neurons_in_plot = min(5, max(neurons_l1-idx_plot*5,0))
                
            # Start creating the Figure
            time_arr = np.arange(0,sim_time,time_step)
            axis1 = plt.figure(layout="constrained").subplot_mosaic(
                [
                    ["0,0","0,1","0,2","0,3"],
                    ["1,0","1,1","1,2","1,3"],
                    ["2,0","2,1","2,2","2,3"],
                    ["3,0","3,1","3,2","3,3"],
                    ["4,0","4,1","4,2","4,3"],
                    ["input","input","input","input"]
                ],
                sharex=True
            )

            # Loop over columns and rows of each plot
            for column in range(4):
                first_neuron_in_plot = 0+5*idx_plot
                neuron = first_neuron_in_plot
                for row in range(max(l0_neurons_in_plot, l1_neurons_in_plot)):
                    
                    if column ==0:  # mem_pot shape (steps, neurons)
                        if neuron> neurons_l0-1: break
                        y = l0_mem_pot[:,neuron]

                    if column ==1:  # spike count shape (neurons, steps)
                        if neuron> neurons_l0-1: break
                        y= l0_spike_count[neuron,:]

                    if column ==2:
                        if neuron> neurons_l1-1: break
                        y = l1_mem_pot[:,neuron]
                        
                    # Plot in the second column
                    if column ==3:
                        if neuron> neurons_l1-1: break
                        y=l1_spike_count[neuron,:]
                    
                    if neuron == first_neuron_in_plot:
                        if column == 0: title_plot = "L0 membrame potential"
                        if column == 1: title_plot = "L0 Spike Count"
                        if column == 2: title_plot = "L1 membrame potential"
                        if column == 3: title_plot = "L2 Spike Count"
                        axis1[str(row)+","+str(column)].set_title(title_plot)
                    
                    if column ==2:
                        axis1[str(row)+","+str(column)].set_ylabel(str(neuron+1), rotation=0, fontsize= 15, labelpad=13)

                    axis1[str(row)+","+str(column)].plot(time_arr,y)
                    axis1[str(row)+","+str(column)].xaxis.grid()

                    ### only plot in Mem Pot plots
                    if column ==0 or column ==2:
                        # Show the threshold 
                        if column == 2 and config["LAYER_SETTING"]["l1"]["adaptive"]:
                            t = l1_thres[:,:]
                            base_t = all_parameters["l1.neuron.base_t"].detach().numpy()[neuron]
                            add_t = all_parameters["l1.neuron.add_t"].detach().numpy()[:]
                            
                            if config["LAYER_SETTING"]["l1"]["adapt_2x2_connection"]: #t = (1,N) self.add_t = (N,N)
                                t = torch.Tensor(t)
                                add_t = torch.Tensor(add_t)
                                threshold = base_t +torch.matmul(t, add_t).detach().numpy()[:,neuron]
                            else:                      #t = (1,N) self.add_t = (N)
                                threshold = base_t +add_t * t

                            axis1[str(row)+","+str(column)].plot(time_arr, threshold, color="r")
                        else:
                            if column == 0:
                                threshold = all_parameters["l0.neuron.thresh"].detach().numpy()[neuron]
                            if column == 2:
                                threshold = all_parameters["l1.neuron.thresh"].detach().numpy()[neuron]   

                            axis1[str(row)+","+str(column)].axhline(threshold,color="r")

                        # Plot the different background, corresponding with target sign
                        if colored_background == True:
                            for i in range(len(pos_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(pos_idx_start[i]*time_step, pos_idx_end[i]*time_step, facecolor="g", alpha= 0.2)
                            for i in range(len(neg_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(neg_idx_start[i]*time_step, neg_idx_end[i]*time_step, facecolor="r", alpha= 0.2)
                            for i in range(len(zero_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(zero_idx_start[i]*time_step, zero_idx_end[i]*time_step, facecolor="k", alpha= 0.2)

                    ### only plot in Spike Count plots
                    if column ==1 or column ==3:
                        #Bound the spike count plot between 0 and 1
                        axis1[str(row)+","+str(column)].set_ylim(-0.05,1.1)
                        
                        # Plot the different background, corresponding with target sign
                        if colored_background == True and spike_count_plot==True:
                            for i in range(len(pos_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(pos_idx_start[i]*time_step, pos_idx_end[i]*time_step, facecolor="g", alpha= 0.2)
                            for i in range(len(neg_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(neg_idx_start[i]*time_step, neg_idx_end[i]*time_step, facecolor="r", alpha= 0.2)
                            for i in range(len(zero_idx_start)):
                                axis1[str(row)+","+str(column)].axvspan(zero_idx_start[i]*time_step, zero_idx_end[i]*time_step, facecolor="k", alpha= 0.2)
                    neuron = neuron +1
            time_arr = np.arange(0,sim_time,time_step)

            ### Plot the lowest figure
            axis1["input"].plot(time_arr,error_arr, label = "SNN input")
            axis1["input"].plot(time_arr,state_l2_arr, label = "SNN output")
            axis1["input"].plot(time_arr,ideal_pid_response, label = "PID reponse")
            axis1["input"].axhline(0,linestyle="--", color="k")
            axis1["input"].xaxis.grid()
        plt.legend()


    # ### Plot the lowest figure
    # plt.figure()
    # plt.plot(time_arr,error_arr, label = "SNN input")
    # plt.plot(time_arr,state_l2_arr, label = "SNN output")
    # plt.plot(time_arr,ideal_pid_response, label = "PID reponse")
    # plt.axhline(0,linestyle="--", color="k")
    # plt.grid()
    # plt.title("Controller response")
    # plt.legend()

    # Otherwise the plot and table are shown on the same moment
    if create_table == False and muliple_test_runs_error_plot==False:
        plt.show()

if create_table == True:
    round_digits = 2
    neurons_l0_total = config["LAYER_SETTING"]["l0"]["neurons"]
    neurons_l0_total = config["NEURONS"]        #//TODO remove this line!!!
    neurons_l1_total = config["NEURONS"]

    # Colors for highlighting
    color_mix = {'white':'#FFFFFF','gray': '#D3D3D3','black':'#313639','purple':'#AD688E','orange':'#D18F77','yellow':'#E8E190','ltgreen':'#CCD9C7','dkgreen':'#96ABA0','red':'#FFCCCB',}
    
    #create table for L0 parameters
    if encoding_layer:
        # Add neuron numbers to data
        l0_neurons = np.array([])
        for neur in range(0,neurons_l0_total):
            if neur not in excluded_neurons:
                l0_neurons = np.append(l0_neurons,str(neur+1))
        l0_data = l0_neurons
        l0_column_label = ["Neuron"]

        #Add spike count to the data
        spike_count = np.array([],dtype=int)
        for neuron in range(neurons_l0_total):
            spikes = int(np.sum(l0_spikes[:,neuron]))
            spike_count = np.append(spike_count,spikes)
        l0_data = np.vstack((l0_data,spike_count))
        l0_column_label.append("Spike count")

        # Add all parameters of l0 to the data array
        for parameter in all_parameters.keys():
            if parameter.split(".")[0] != "l0":
                continue
            data_param = np.round(torch.flatten(all_parameters[parameter]).detach().numpy(),round_digits)
            # Only add the parameters which has one param per neurons( so not leak l2 and rec connections)
            if data_param.size == l0_neurons.size:
                l0_data =np.vstack((l0_data,data_param))
                l0_column_label.append(parameter)
            else: print("Parameter named ", parameter, " is not included in the table")
        
        ### Convert to "data" array to a table
        fig = plt.figure(linewidth=1, tight_layout={"pad":1})
        table = plt.table(cellText=np.transpose(l0_data), colLabels=l0_column_label, loc='center')

        # Set font size
        table.auto_set_font_size(False)
        table.set_fontsize(15)

        # Hide axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.suptitle("L0 parameters")

        # Hide axes border
        plt.box(on=None)

        # highlight w1 cells
        idx=1
        for w in l0_data[2,:]:
            w = float(w)
            if float(w)<0:                                      #Since the items in data are strings
                table[idx,2].set_facecolor(color_mix["red"])
            idx = idx+1
    ##########          end of table l0             ############

    #create table for L1 parameters
    # Add neuron numbers to data
    l1_neurons = np.array([])
    for neur in range(0,neurons_l1_total):
        if neur not in excluded_neurons:
            l1_neurons = np.append(l1_neurons,str(neur+1))
    l1_data = l1_neurons
    l1_column_label = ["Neuron"]

    #Add spike count to the data
    spike_count = np.array([],dtype=int)
    for neuron in range(neurons_l1_total):
        spikes = int(np.sum(l1_spikes[:,neuron]))
        spike_count = np.append(spike_count,spikes)
    l1_data = np.vstack((l1_data,spike_count))
    l1_column_label.append("Spike count")

    # Add all parameters of l0 to the data array
    param_not_in_main_table =[]
    for parameter in all_parameters.keys():
        if parameter.split(".")[0] != "l1" and parameter != "l2.ff.weight":
            continue
        data_param = np.round(torch.flatten(all_parameters[parameter]).detach().numpy(),round_digits)
        # Only add the parameters which has one param per neurons( so not leak l2 and rec connections)
        if data_param.size == l1_neurons.size:
            l1_data =np.vstack((l1_data,data_param))
            l1_column_label.append(parameter)
        else: 
            print("Parameter named ", parameter, " is not included in the table")
            param_not_in_main_table.append(parameter)
    
    if "l1.ff.weight" not in param_not_in_main_table:
        #find row with w1 and w2 and swap them such they are in the beginning of the table
        ind_w1 = l1_column_label.index("l1.ff.weight")
        l1_data[[ind_w1,2]] = l1_data[[2,ind_w1]]
        l1_column_label[2], l1_column_label[ind_w1] = l1_column_label[ind_w1], l1_column_label[2]

        ind_w2 = l1_column_label.index("l2.ff.weight")
        l1_column_label[3], l1_column_label[ind_w2] = l1_column_label[ind_w2], l1_column_label[3]     # Swap the w1 to 3 row
        l1_data[[ind_w2,3]] = l1_data[[3,ind_w2]]     
    else:
        ind_w2 = l1_column_label.index("l2.ff.weight")
        l1_column_label[2], l1_column_label[ind_w2] = l1_column_label[ind_w2], l1_column_label[2]     # Swap the w1 to 3 row
        l1_data[[ind_w2,2]] = l1_data[[2,ind_w2]]  


    ### Convert to "data" array to a table
    fig = plt.figure(linewidth=1, tight_layout={"pad":1})
    table = plt.table(cellText=np.transpose(l1_data), colLabels=l1_column_label, loc='center')

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(15)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.suptitle("L1 & L2 parameters")

    # Hide axes border
    plt.box(on=None)



                                          # Swap w2 to row 4
    
    # highlight weight cells in third column
    idx=1
    for w in l1_data[2,:]:
        w = float(w)
        if float(w)<0:                                      #Since the items in data are strings
            table[idx,2].set_facecolor(color_mix["red"])
        idx = idx+1

    if "l1.ff.weight" not in param_not_in_main_table:
        # highlight w1 cells
        idx=1
        for w in l1_data[3,:]:
            w = float(w)
            if float(w)<0:                                      #Since the items in data are strings
                table[idx,3].set_facecolor(color_mix["red"])
            idx = idx+1

    ######          End main table l1           #######
    for param in param_not_in_main_table:
        if param == "l1.ff.weight":
            data_l1_ff_weight = np.round(all_parameters[param].detach().numpy(),round_digits)
            neur_l0_list = []
            neur_l1_list = []
            for i in range(1,neurons_l0_total+1): neur_l0_list.append(str(i))
            for i in range(1,neurons_l1_total+1): neur_l1_list.append(str(i))

            fig = plt.figure(linewidth=1, tight_layout={"pad":1})
            table = plt.table(cellText=data_l1_ff_weight, colLabels=neur_l0_list, rowLabels=neur_l1_list, loc='center')

            # Set font size
            table.auto_set_font_size(False)
            table.set_fontsize(13)

            # Hide axes
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            fig.suptitle("L1 ff Weight (L1 neurons, L0 neurons)")

            # Hide axes border
            plt.box(on=None)

            # highlight negative cells
            for row in range(len(neur_l1_list)):
                for column in range(len(neur_l0_list)):
                    w = float(data_l1_ff_weight[row,column])
                    if float(w)<0:                                     
                        table[row,column].set_facecolor(color_mix["red"])


            plt.show()

        if param == "l1.rec.weight":
            data_l1_rec_weight = np.round(all_parameters[param].detach().numpy(),round_digits)

            neur_l1_list = []
            for i in range(1,neurons_l1_total+1): neur_l1_list.append(str(i))

            fig = plt.figure(linewidth=1, tight_layout={"pad":1})
            table = plt.table(cellText=data_l1_rec_weight, colLabels=neur_l1_list, rowLabels=neur_l1_list, loc='center')

            # Set font size
            table.auto_set_font_size(False)
            table.set_fontsize(13)

            # Hide axes
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            fig.suptitle("L1 Recurrent Weight (To, From)")

            # Hide axes border
            plt.box(on=None)

            # highlight negative cells
            for row in range(len(neur_l1_list)):
                for column in range(len(neur_l0_list)):
                    w = float(data_l1_rec_weight[row,column])
                    if float(w)<0:                                     
                        table[row+1,column].set_facecolor(color_mix["red"])

            plt.show()
    ####      End plots Rec and FF weight   ###########


    
# if create_table_old == True:
#     ######################### plot table ##########################
#     round_digits = 3

#     # Add neuron numbers to data
#     neurons = np.array([])
#     for neur in range(0,config["NEURONS"]):
#         if neur not in excluded_neurons:
#             neurons = np.append(neurons,str(neur+1))
#     data = neurons

#     #Add spike count to the data
#     spike_count = np.array([],dtype=int)
#     for neuron in range(number_of_neurons):
#         spikes = int(np.sum(l1_spikes[:,neuron]))
#         spike_count = np.append(spike_count,spikes)
#     data = np.vstack((data,spike_count))

#     column_label = ["Neuron", "Spike count"]

#     # Add all parameters to the data array
#     for parameter in all_parameters.keys():
#         data_param = np.round(torch.flatten(all_parameters[parameter]).detach().numpy(),round_digits)
#         # Only add the parameters which has one param per neurons( so not leak l2 and rec connections)
#         if data_param.size == neurons.size:
#             data =np.vstack((data,data_param))
#             column_label.append(parameter)
#         else: print("Parameter named ", parameter, " is not included in the table")

#     #find row with w1 and w2 and swap them such they are in the beginning of the table
#     ind_w1 = column_label.index("l1.ff.weight")
#     ind_w2 = column_label.index("l2.ff.weight")
#     data[[ind_w1,2]] = data[[2,ind_w1]]
#     column_label[3], column_label[ind_w2] = column_label[ind_w2], column_label[3]     # Swap the w1 to 3 row
#     data[[ind_w2,3]] = data[[3,ind_w2]]                                               # Swap w2 to row 4
#     column_label[2], column_label[ind_w1] = column_label[ind_w1], column_label[2]

#     # Add "Impact" score in table (w2*spike count)
#     w2 = all_parameters["l2.ff.weight"].detach().numpy()
#     impact_abs = np.abs(w2*spike_count)
#     impact_norm = np.round(impact_abs/np.max(impact_abs)*10,2)
#     column_label = np.insert(column_label,1,"Impact")
#     data = np.insert(data,1,impact_norm,axis=0)
    
#     ### Convert to "data" array to a table
#     plt.figure(linewidth=1,
#             tight_layout={"pad":1})
#     table = plt.table(cellText=np.transpose(data), colLabels=column_label, loc='center')

#     # Set font size
#     table.auto_set_font_size(False)
#     table.set_fontsize(15)

#     # Hide axes
#     ax = plt.gca()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

#     # Hide axes border
#     plt.box(on=None)

#     # Colors for highlighting
#     color_mix = {'white':'#FFFFFF','gray': '#D3D3D3','black':'#313639','purple':'#AD688E','orange':'#D18F77','yellow':'#E8E190','ltgreen':'#CCD9C7','dkgreen':'#96ABA0','red':'#FFCCCB',}

#     # highlight w1 cells
#     idx=1
#     for w in data[3,:]:
#         w = float(w)
#         if float(w)<0:                                      #Since the items in data are strings
#             table[idx,3].set_facecolor(color_mix["red"])
#         idx = idx+1

#     idx=1
#     for w in data[4,:]:
#         if float(w)<0:
#             table[idx,4].set_facecolor(color_mix["red"])
#         idx = idx+1

#     #greyout the non spiking neurons (NOTE: data is later inversed, sot it is column,row now)
#     row=1
#     for spike_count in data[2,:]:
#         if int(spike_count) == 0:
#             for col in range(len(data)):
#                 table[row,col].set_facecolor(color_mix["gray"])
#         row +=1


#     if "l1.rec.weight" not in all_parameters:
#         plt.show()

#     # Create the recurrent table
#     if "l1.rec.weight" in all_parameters:
#         # data = neurons[..., np.newaxis]
#         data = np.round(all_parameters["l1.rec.weight"].detach().numpy(),round_digits)

#         norm = plt.Normalize(data.min(), data.max())
#         colours = plt.cm.RdYlGn(norm(data))

#         column_label = [str(i) for i in range(1,number_of_neurons+1)]
#         plt.figure(linewidth=1,
#                 tight_layout={"pad":1})
#         table = plt.table(cellText=data, colLabels=column_label, rowLabels=column_label, cellColours=colours, loc='center')

#         # Set font size
#         table.auto_set_font_size(False)
#         table.set_fontsize(15)

#         # Hide axes
#         ax = plt.gca()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#         # Hide axes border
#         plt.box(on=None)
#         plt.show()


if muliple_test_runs_error_plot == True:
    plt.figure()
    gen_arr = np.arange(0,len(solutions_error)*step_size,step_size)
    # if generations%step_size!=0:
    #     gen_arr = np.append(gen_arr,generations)
    plt.plot(gen_arr,solutions_error)
    plt.yscale('log')
    plt.title("Error of Manual Test Run every "+ str(step_size) +" generations")
    plt.xlabel("Generations [-]")
    plt.ylabel("Error [-]")
    plt.grid()
    plt.show()



# if create_csv_file == True:
#     ### Create a matlab file with parameters
#     data_mat = np.transpose(np.vstack((w1,w2,thres,a,b,c,d,v2,v1,v0,utau)))
#     np.savetxt("test_matlab.csv", data_mat)

# if spectal_analysis == True:
#     N = len(predictions)
#     T= 1./500.
#     x = np.linspace(0,N*T,N,endpoint=False)
#     y = predictions

#     yf= fft(y)
#     xf= fftfreq(N,T)[:N//2]

#     plt.figure()
#     plt.plot(xf,2.0/N*np.abs(yf[0:N//2]))
#     plt.grid()
#     plt.show()




# if plot_sigma == True:
#     sigma = dict_solutions["sigma"]
#     gen_arr = np.arange(0, len(sigma))
#     plt.figure()
#     plt.title("Stepsize over generations")
#     plt.xlabel("Generations [-]")
#     plt.ylabel("Stepsize [-]")
#     plt.plot(gen_arr,sigma)
#     plt.grid()
#     plt.show()

# if plot_parameters_evolution == True:

#     generations = np.size(dict_solutions["mean"],0)
#     parameters_mean = dict_solutions["mean"] #shape (generations, parameters)
#     final_parameters =torchga.model_weights_as_dict(model, parameters_mean[0,:])

#     ## Initialize dictonary structre{param: [params_gen1, params_gen2 etc..].}
#     full_solution_dict = {key:None for key, _ in final_parameters.items()}



#     ### Fill the dictornary
#     for param in full_solution_dict:
#         for gen in range(generations):
#             gen_parameters =torchga.model_weights_as_dict(model, parameters_mean[gen,:])
#             for name, value in gen_parameters.items():
#                 if param == name:
#                     value = torch.flatten(value).detach().numpy()
#                     if full_solution_dict[param] is None: #Check is dict is empty 
#                         full_solution_dict[param] = value
#                     else:
#                         full_solution_dict[param] = np.vstack((full_solution_dict[param],value))

#                     break


#     ### Plot the different diagrams
#     gen_arr = np.arange(0,generations)
#     for param in full_solution_dict:
#         plt.figure()
#         for num_param in range(len(full_solution_dict[param][0,:])):
#             param_per_gen = full_solution_dict[param][:,num_param]
#             param_per_gen = param_per_gen.flatten()
#             plt.plot(gen_arr,param_per_gen,label=str(num_param))
#         plt.title("Evolution of " + str(param))
#         plt.xlabel("Generations [-]")
#         plt.xticks(gen_arr)
#         plt.legend()
#         plt.grid()
#     plt.show()
        
                    
