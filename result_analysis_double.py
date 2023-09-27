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
from LIF_EVOTORCH import get_dataset , evaluate_fitness, run_controller_double
import copy

# for dataset_number in range(10):
sim_time = 300
dataset_number = None                                                  # None is the test_dataset
folder_models = "Results_EA/Blimp/10Hz"                                               # all folder under the folder Results_EA
window_size =6      #NOTE: even numbers
#config["DATASET_DIR"] = "Sim_data/height_control_PID/fast_steps"


solution_type                   = "best_test"         #"best_test" = best performance on manual dataset     "best_overall" = best performance overall (can be easy dataset) "last" = last solution
filename_pd = 228
filename_i = 247





# Run the simulation
def run_sim(fitness_mode, config, controller,solution,input_dataset):
    #The solution is defined in a numpy array shape
    sys_output = None

    #Init PD controller
    controller_pd = controller[0]
    solution_pd = solution[0]
    final_parameters_pd =torchga.model_weights_as_dict(controller_pd,solution_pd )
    controller_pd.load_state_dict(final_parameters_pd)

    controller_pd.l0.init_reshape()
    controller_pd.l1.init_reshape()
    controller_pd.l2.init_reshape()

    #Init i controller
    controller_i = controller[1]
    solution_i = solution[1]
    final_parameters_pd =torchga.model_weights_as_dict(controller_i,solution_i )
    controller_i.load_state_dict(final_parameters_pd)

    controller_i.l0.init_reshape()
    controller_i.l1.init_reshape()
    controller_i.l2.init_reshape()

    if  fitness_mode== 1: #Only controller simlation
        state_l2_arr= run_controller_double(controller_pd, controller_i,input_dataset)
        fitness_measured = state_l2_arr
        error_arr = torch.flatten(input_dataset).detach().numpy()

    return fitness_measured, error_arr

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

def get_solution(folder_models, filename,solution_type):
    #Pick last file name if filename is None
    if filename == None:
        all_files = os.listdir(folder_models)
        splitted_files = [int(f.split("-")[0]) for f in all_files]
        max_ind = splitted_files.index(max(splitted_files))
        filename = all_files[max_ind].split(".")[0] #get rid of .pkl
        print("\nFilename used = ", filename)

    # Pick the filename number
    if type(filename) == int:
        all_files = os.listdir(folder_models)
        splitted_files = [int(f.split("-")[0]) for f in all_files]
        index_file = splitted_files.index(filename)
        filename = all_files[index_file].split(".")[0] #get rid of .pkl
        print("\nFilename used = ", filename)


    pickle_in = open(folder_models +"/" + filename+".pkl","rb")
    dict_solutions = pickle.load(pickle_in)
    solution = dict_solutions["best_solution"]
    solutions_error = dict_solutions["error"]
    step_size = dict_solutions["step_size"]
    config = dict_solutions["config"]

    if solution_type == "best_test":
        solutions = dict_solutions["test_solutions"]
        
        
        generations = dict_solutions["generations"]

        best_sol_ind = np.argmin(solutions_error)
        solution = solutions[best_sol_ind+1] #+1 since first of solution_testrun is only zeros
        if solution_type == "last":
            solution = solutions[-1] #+1 since first of solution_testrun is only zeros
            print("Plot shown with last generation")

        # Check if the best solution is the last generation
        if best_sol_ind == len(solutions_error)-1:
            best_gen = generations
        else: best_gen = (best_sol_ind)*step_size
        if solution_type != "last": print("Best solutions of the intermidiate testrun implementation is found at ", best_gen, " generations")
    else: print("Solution: Best evaluated solution (note: can be the result of an easy training dataset)")

    return solution, dict_solutions, config

#########################################################################################

solution_pd, dict_pd, config_pd = get_solution(folder_models, filename_pd, solution_type)
solution_i, dict_i, config_i = get_solution(folder_models, filename_i, solution_type)

# Init PD controller
if config_pd["LAYER_SETTING"]["l0"]["enabled"]: controller_pd = Encoding_L1_Decoding_SNN(None, config_pd["NEURONS"], config_pd["LAYER_SETTING"])
else:                                           controller_pd = L1_Decoding_SNN(None, config_pd["NEURONS"], config_pd["LAYER_SETTING"])

# Init I controller
if config_i["LAYER_SETTING"]["l0"]["enabled"]: controller_i = Encoding_L1_Decoding_SNN(None, config_i["NEURONS"], config_i["LAYER_SETTING"])
else:                                          controller_i = L1_Decoding_SNN(None, config_i["NEURONS"], config_i["LAYER_SETTING"])



# Initialize varibales from problem Class
print("\nDataset used = ", config_pd["DATASET_DIR"], "\nDatasetnumber = ", dataset_number)
config_pd["ALTERNATIVE_TARGET_COLUMN"] = [1]

input_data, fitness_target = get_dataset(config_pd, dataset_number, sim_time)
fitness_mode = config_pd["TARGET_FITNESS"]


##################           RUN SIM                ########################################################
fitness_measured, error_arr = run_sim(fitness_mode, [config_pd,config_i],[controller_pd, controller_i],[solution_pd, solution_i],input_data)




# Calculate the fitness value
fitness_value = evaluate_fitness(config_pd["FITNESS_FUNCTION"], fitness_measured, fitness_target)
print("Fitness value = ", np.round(fitness_value.item(),5))


if fitness_mode == 1:   label_fitness_measured = "SNN output"; label_fitness_target = "PID output"


title = "Controller Response"
time_test = np.arange(0,sim_time,config_pd["TIME_STEP"])


plt.plot(time_test, fitness_measured, color = "b", label=label_fitness_measured)
plt.plot(time_test, fitness_target, color = 'r',label=label_fitness_target)
plt.plot(time_test, error_arr, label = "Error" )
plt.title(title)
plt.grid()
plt.legend()

plt.show()
