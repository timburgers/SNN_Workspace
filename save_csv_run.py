import torch
import pygad.torchga as torchga
from SNN_LIF_LI_init import L1_Decoding_SNN, Encoding_L1_Decoding_SNN
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
from LIF_EVOTORCH import get_dataset, run_controller, run_controller_dynamics, evaluate_fitness
import pandas as pd


# for dataset_number in range(10):
sim_time = 500
dataset_number = None                                                  # None is the test_dataset
file_list = [119]                                                       #None --> highest number, or int or str (withou .pkl)
folder_of_model = "Simulation/I"                                               # all folder under the folder Results_EA
use_alternative_dataset = None #"Sim_data/height_control_PID/pos_slope_then_zero"
use_alternative_input = None
use_alternative_target = None
only_controller_input = True                                #set the fitness function to 1
add_moving_averge = True

show_plots = True
save_csv = True
folder_saved_csv = "test"
plot_with_best_testrun          = False  #True: solution = best performance on manual dataset      False: solution = best performance overall (can be easy dataset)
plot_last_generation            = False

for filename in file_list:
    if os.path.isdir("Results_EA/"+folder_of_model+"/"+folder_saved_csv): pass
    else: os.mkdir("Results_EA/"+folder_of_model+"/"+folder_saved_csv)

    #Pick last file name
    if filename == None:
        all_files = os.listdir("Results_EA/"+folder_of_model)
        all_files = [f for f in all_files if f.endswith(".pkl")]
        splitted_files = [int(f.split("-")[0]) for f in all_files]
        max_ind = splitted_files.index(max(splitted_files))
        filename = all_files[max_ind].split(".")[0] #get rid of .pkl
        print("\nFilename used = ", filename)

    if type(filename) == int:
        all_files = os.listdir("Results_EA/"+folder_of_model)
        all_files = [f for f in all_files if f.endswith(".pkl")]
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



    config = dict_solutions["config"]
    number_of_neurons = config["NEURONS"]
    #//TODO GET rid of beun fixes here
    # config["LAYER_SETTING"]["l0"]["shared_leak_iv"] = False
    # config["LAYER_SETTING"]["l1"]["shared_leak_iv"] = False
    # config["LAYER_SETTING"]["l0"]["shared_thres"] = False
    # config["LAYER_SETTING"]["l1"]["shared_2x2_weight_cross"] = False
    # config["LAYER_SETTING"]["l1"]["adapt_share_baseleak_t"] = False
    # config["LAYER_SETTING"]["l1"]["recurrent_2x2"] = False
    if use_alternative_dataset != None:
        config["DATASET_DIR"]= use_alternative_dataset 

    if use_alternative_input != None:
        config["ALTERNATIVE_INPUT_COLUMN"]= use_alternative_input 
    if use_alternative_target != None:
        config["ALTERNATIVE_TARGET_COLUMN"]= use_alternative_target 
    try: _ = config["LAYER_SETTING"]["l1"]["recurrent_2x2"]
    except: config["LAYER_SETTING"]["l1"]["recurrent_2x2"] = False


    encoding_layer = config["LAYER_SETTING"]["l0"]["enabled"]
    if encoding_layer: controller = Encoding_L1_Decoding_SNN(None, config["NEURONS"], config["LAYER_SETTING"])
    else:              controller = L1_Decoding_SNN(None, config["NEURONS"], config["LAYER_SETTING"])

    #//TODO GET rid of beun fixes here
    controller.fitness_func = "mse"



    time_step = config["TIME_STEP"]

    total_time_steps = int(sim_time/time_step)

    # Initialize varibales from problem Class
    print("\nDataset used = ", config["DATASET_DIR"], "\nDatasetnumber = ", dataset_number)
    input_data, fitness_target = get_dataset(config, dataset_number, sim_time)
    if only_controller_input:
        fitness_mode = 1
    else:
        fitness_mode = config["TARGET_FITNESS"]


    ##################           RUN SIM                ########################################################
    fitness_measured, error_arr, state_l0_arr ,state_l1_arr ,state_l2_arr ,trained_parameters = run_sim(fitness_mode, config,controller,solution,input_data,True)
    all_parameters = controller.state_dict()       #Convert the dict of only trained paramaters, to a dict with all parameters per neuron
    l1_current,l1_mem_pot, l1_thres,l1_spikes = set_variables(config["LAYER_SETTING"]["l1"]["adaptive"],state_l1_arr)
    l0_current,l0_mem_pot, _ ,l0_spikes = set_variables(False,state_l0_arr)

    config["TARGET_FITNESS"] = 1 #since the target of mode 1 is the pid response
    _, ideal_pid_response = get_dataset(config, dataset_number, sim_time)



    # Calculate the fitness value
    fitness_value = evaluate_fitness(controller.fitness_func, fitness_measured, fitness_target, True)
    print("Fitness value = ", np.round(fitness_value.item(),5))
    time_test = np.arange(0,sim_time,config["TIME_STEP"])
    if show_plots:
        if fitness_mode == 1:   label_fitness_measured = "SNN output"; label_fitness_target = "PID output"
        if fitness_mode == 2:   label_fitness_measured = "Blimp Height SNN"; label_fitness_target = "Blimp Height Reference"
        if fitness_mode == 3:   label_fitness_measured = "Blimp Height SNN"; label_fitness_target = "Blimp height PID"

        title = "Controller Response"
        
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


        plt.figure()
        plt.title("Controller response")
        plt.plot(time_test, state_l2_arr, label = "Controller Output")
        plt.plot(time_test, ideal_pid_response, label = "PID Output")
        plt.legend()

        plt.show()

    if save_csv:
        df_final = pd.DataFrame(columns=["time","ref","meas","u","target_h"])
        df_final["time"] = time_test
        df_final["error"] = torch.flatten(input_data)
        df_final["meas"] = fitness_measured
        df_final["u"] = state_l2_arr
        df_final["Target"] = fitness_target

        df_final.to_csv(path_or_buf= "Results_EA/"+ folder_of_model +"/"+folder_saved_csv+"/" + filename + ".csv", index=False)






