
#!/usr/bin/python3
import torch
import pygad.torchga as torchga
from SNN_LIF_LI_init import L1_Decoding_SNN, Encoding_L1_Decoding_SNN
from wandb_log_functions import number_first_wandb_name
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
# os.environ["RAY_DEDUP_LOGS"]= "0"
from random import randint
from torchmetrics import PearsonCorrCoef

from evotorch import Problem
from evotorch.algorithms import CMAES, PyCMAES
from evotorch.logging import StdOutLogger,WandbLogger
import ray
import wandb
from datetime import datetime
import time
import platform
import math
import sys, getopt
from sim_dynamics.Dynamics import Blimp
import copy
import random
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
ray.init(log_to_driver=False, include_dashboard=False)


class LIF_EA_evotorch(Problem):
    def __init__(self, config_path):
        if platform.system() == "Linux":
            
            # Check if running on pc or delftblue
            if os.path.isdir("/home/tim"):
                self.prefix = "/home/tim/SNN_Workspace/"
            else: self.prefix = "/scratch/timburgers/SNN_Workspace/"


        if platform.system() == "Windows":
            self.prefix = ""

        ### Read config file
        with open(self.prefix + config_path + ".yaml","r") as f:
            self.config = yaml.safe_load(f)
        self.config = fix_requirements_in_config(self.config)
        show_layer_settings(self.config)

        # Select model
        self.encoding_layer = self.config["LAYER_SETTING"]["l0"]["enabled"]
        if self.encoding_layer: self.model = Encoding_L1_Decoding_SNN(None, self.config["NEURONS"], self.config["LAYER_SETTING"])
        else:                   self.model = L1_Decoding_SNN(None, self.config["NEURONS"], self.config["LAYER_SETTING"])
        self.number_parameters = 0

        # Log the structure of the model that is used
        self.dict_model_structure = dict()
        for name, parameter in self.model.named_parameters():
            self.number_parameters = self.number_parameters + torch.flatten(parameter).shape[0]
            self.dict_model_structure[name] = torch.flatten(parameter).shape[0]

        
        self.input_data, self.target_data = get_dataset(self.config, self.config["DATASET_NUMBER"], self.config["SIM_TIME"])
        self.bounds = create_bounds(self, self.model, self.config)
        self.prefix = ""
        self.all_param_test_solutions = dict()
        self.test_solutions= np.zeros(self.number_parameters)
        self.error_test_solutions= np.array([])
        self.train_datasets=np.array([])
        self.manual_dataset_prev = False
        self.fitness_mode = self.config["TARGET_FITNESS"]
        self.fitness_func = self.config["FITNESS_FUNCTION"]

        # Get the number of datasets (with only the word 'datatset' before the _)
        all_files_in_folder = os.listdir(self.prefix + self.config["DATASET_DIR"])
        splitted_files = [f.split("_")[0] for f in all_files_in_folder]
        self.max_number_of_datasets = splitted_files.count("dataset")

        self.dataset = None
        self.input_data_new = None
        self.target_data_new = None

        # self.C_matrix = None
        self.mean = None
        self.sigma = None
        self.stds = None
        
        super().__init__(
            objective_sense="min",
            solution_length=self.number_parameters,
            initial_bounds=(-1,1),
            num_actors=self.config["NUMBER_PROCESSES"],
            bounds=self.bounds,
        )

    def _evaluate(self, solution):
        solution_np = solution.values.detach().numpy() #numpy array of all parameters
        controller = copy.deepcopy(self.model)
        final_parameters =torchga.model_weights_as_dict(controller, solution_np)
        controller.load_state_dict(final_parameters)

        # Convert the final parameters in the correct shape
        if self.config["LAYER_SETTING"]["l0"]["enabled"]: controller.l0.init_reshape()
        controller.l1.init_reshape()
        controller.l2.init_reshape()

        if  self.fitness_mode== 1: #Only controller simlation
            fitness_measured = run_controller(controller,self.input_data, save_mode = False)
        elif self.fitness_mode == 2 or self.fitness_mode == 3:#Also simulate the dyanmics
            fitness_measured = run_controller_dynamics(self.config,controller,self.input_data, save_mode = False)

        # Calculate the fitness value
        fitness_value = evaluate_fitness(self.fitness_func, fitness_measured, self.target_data)

        print(fitness_value)
        solution.set_evals(fitness_value)

def fix_requirements_in_config(config):
    l0 = config["LAYER_SETTING"]["l0"]
    l1 = config["LAYER_SETTING"]["l1"]
    l2 = config["LAYER_SETTING"]["l2"]
    
    # Check when l0-l1 is same number of neurons
    if l0["enabled"] and (l0["neurons"] == config["NEURONS"] or l1["w_diagonal"]):
        l0_l1_square = True
        print("\nL0 and L1 have the same number of neurons")
    else:l0_l1_square = False

    if l0["enabled"] == False:
        config["LAYER_SETTING"]["l0"]["neurons"] = None
        config["LAYER_SETTING"]["l0"]["bias"] = False
        config["LAYER_SETTING"]["l0"]["shared_weight_and_bias"] = False
        config["LAYER_SETTING"]["l0"]["shared_leak_i"] = False
        config["LAYER_SETTING"]["l0"]["clamp_v"] = False

    if l0_l1_square and l1["w_diagonal"]:
        config["LAYER_SETTING"]["l0"]["neurons"] = config["NEURONS"]
    
    if l0_l1_square == False or l1["adaptive"] == False:
        config["LAYER_SETTING"]["l1"]["adapt_thres_input_spikes"] = False
        config["LAYER_SETTING"]["l1"]["adapt_2x2_connection"] = False
        config["LAYER_SETTING"]["l1"]["adapt_share_add_t"] = False
        print("\nSet all adapt parameters to False since l0-l1 is not squared")
    
    if l0["enabled"] and l1["w_diagonal"] == False:
        config["LAYER_SETTING"]["l1"]["w_diagonal_2x2"] = False
        print("\nSet w_diagonal_2x2 to False since l0-l1 is not diagonal")    
    return config

def show_layer_settings(config):
    lay_set = config["LAYER_SETTING"]
    neurons_l1 = config["NEURONS"]
    if lay_set["l1"]["w_diagonal"]: neurons_l0 = neurons_l1
    else:                           neurons_l0 = lay_set["l0"]["neurons"]

    for layer_name, layer_settings in lay_set.items():

        if  layer_name=="l0":
            if lay_set["l0"]["enabled"]: print("\n--------- Encoding Layer (",neurons_l0,") -----------")
            else: continue

        if layer_name=="l1": print("\n--------- Main Layer (",neurons_l1,")-----------")
        if layer_name=="l2": print("\n--------- Decoding Layer (1) -----------")

        for set_name, set_value in layer_settings.items():
            if set_value == True:
                print(set_name)
        print("------------------------------------------\n")

# def create_wandb_description(config):
#     desciption = ""
#     lay_set = config["LAYER_SETTING"]
#     for layer_name, layer_settings in lay_set.items():
#         for set_name, set_value in layer_settings.items():
#             if set_value == True:


def run_controller(controller,input,save_mode):
    # Initialize neurons states
    if controller.encoding_layer: state_l0 = torch.zeros(controller.l0.neuron.state_size, 1, controller.l1_input)
    state_l1                               = torch.zeros(controller.l1.neuron.state_size, 1, controller.neurons) 
    state_l2                               = torch.zeros(controller.l2.neuron.state_size,1)

    state_l2_arr = np.array([])
    if save_mode == False:
        for t in range(input.shape[1]):
            error = input[:,t,:]

            # Run controller
            if controller.encoding_layer: state_l0, state_l1, state_l2 = controller(error,state_l0, state_l1, state_l2)
            else:                         state_l1, state_l2 = controller(error, state_l1, state_l2)

            # Append states to array
            state_l2_arr = np.append(state_l2_arr,state_l2.detach().numpy())
        return state_l2_arr

    elif save_mode == True:
        # Return None when encodig layer is disabled
        state_l0_arr = None

        for t in range(input.shape[1]):
            error = input[:,t,:]
            
            # Run controller
            if controller.encoding_layer: state_l0, state_l1, state_l2 = controller(error,state_l0, state_l1, state_l2)
            else:                         state_l1, state_l2 = controller(error, state_l1, state_l2)

            # If applicable, append states of l0 to array
            if controller.encoding_layer:
                if t==0: state_l0_arr = state_l0.detach().numpy()[np.newaxis,...]
                else: state_l0_arr = np.concatenate((state_l0_arr, state_l0.detach().numpy()[np.newaxis, ...]))       # Append in this manner to conserve state

            # Append states of l1 to array
            if t==0: state_l1_arr = state_l1.detach().numpy()[np.newaxis,...]
            else: state_l1_arr = np.concatenate((state_l1_arr, state_l1.detach().numpy()[np.newaxis, ...]))       # Append in this manner to conserve state

            # Append states of l2 to array
            state_l2_arr = np.append(state_l2_arr,state_l2.detach().numpy())

        return state_l2_arr, state_l1_arr, state_l0_arr


def run_controller_dynamics(config,controller,input, save_mode):
    # Initialize neurons states
    if controller.encoding_layer: state_l0 = torch.zeros(controller.l0.neuron.state_size, 1, controller.l1_input)
    state_l1                               = torch.zeros(controller.l1.neuron.state_size, 1, controller.neurons) 
    state_l2                               = torch.zeros(controller.l2.neuron.state_size,1)

    #Initilaize dyanmic system
    dyn_system = Blimp(config)

    #Initilaize output arrray
    sys_output = np.array([0])
    
    if save_mode == False:
        for t in range(input.shape[1]):
            ref = input[:,t,:]
            error = ref - sys_output[-1]

            # Run controller
            if controller.encoding_layer: state_l0, state_l1, state_l2 = controller(error,state_l0, state_l1, state_l2)
            else:                         state_l1, state_l2 = controller(error, state_l1, state_l2)

            # Simulate dyanmics
            dyn_system.sim_dynamics(state_l2.detach().numpy())

            #Append states to array
            sys_output = np.append(sys_output,dyn_system.get_z())
        return sys_output[1:] # Skip the first 0 height input

    if save_mode==True:
        # Return None when encodig layer is disabled
        state_l0_arr = None
        error_arr = np.array([])
        state_l2_arr = np.array([])

        for t in range(input.shape[1]):
            ref = input[:,t,:]
            error = ref - sys_output[-1]

            # Run controller
            if controller.encoding_layer: state_l0, state_l1, state_l2 = controller(error,state_l0, state_l1, state_l2)
            else:                         state_l1, state_l2 = controller(error, state_l1, state_l2)

            # Simulate dyanmics
            dyn_system.sim_dynamics(state_l2.detach().numpy())

            # If applicable, append states of l0 to array
            if controller.encoding_layer:
                if t==0: state_l0_arr = state_l0.detach().numpy()[np.newaxis,...]
                else: state_l0_arr = np.concatenate((state_l0_arr, state_l0.detach().numpy()[np.newaxis, ...]))       # Append in this manner to conserve state

            # Append states of l1 to array
            if t==0: state_l1_arr = state_l1.detach().numpy()[np.newaxis,...]
            else: state_l1_arr = np.concatenate((state_l1_arr, state_l1.detach().numpy()[np.newaxis, ...]))

            # Append states of l1 to array
            state_l2_arr = np.append(state_l2_arr, state_l2.detach().numpy())

            error_arr = np.append(error_arr, error.detach().numpy())

            sys_output = np.append(sys_output,dyn_system.get_z())
            
        return sys_output[1:], error_arr, state_l0_arr, state_l1_arr, state_l2_arr # Skip the first 0 height input


def evaluate_fitness(fitness_func, fitness_measured, fitness_target):

    if fitness_func == "mse" or fitness_func == "mse+p": fitt_fn1 = torch.nn.MSELoss()
    if fitness_func == "mae" or fitness_func == "mae+p": fitt_fn1 = torch.nn.L1Loss()


    #Evaluate fitness using MSE and additionally pearson if there should be a linear correlation between target and output
    fitness_target = torch.flatten(fitness_target)
    fitness_measured = torch.from_numpy(fitness_measured)
    fitness_value = fitt_fn1(fitness_measured,fitness_target)

    if fitness_func == "mse+p" or fitness_func == "mae+p":
        fitt_fn2 = PearsonCorrCoef()
        fitness_value += (1-fitt_fn2(fitness_measured,fitness_target)) #pearson of 1 means linear correlation
    return fitness_value
        

# Get the intiail position of the center and of the step size
def init_conditions(problem):

    bounds_config= problem.config["PARAMETER_BOUNDS"] #The bounds are usedto scale the initial step size
    std_init=np.array([])
    model = problem.model
    param_model =torchga.model_weights_as_dict(model, np.ones(problem.number_parameters))

    # Initialize STD (the satandard deviation/stepsize) of the search
    for name, value in param_model.items():
        number_of_params = len(torch.flatten(value).detach().numpy())

        for iteration in range(number_of_params):
            # Fill in initial condition of stepsize (std)
            initial_step_size = (bounds_config[name]["high"]-bounds_config[name]["low"])*problem.config["PERCENT_INTIIAL_STEPSIZE"]
            std_init = np.append(std_init,initial_step_size)


    # Initialize MEAN using a existing solution
    if problem.config["MEAN_SETTING"] == "previous":
        print ("\nManual initialization of the mean with file ", problem.config["PREVIOUS_SOLUTION"])
        pickled_dict = open("Results_EA/Blimp/" + problem.config["PREVIOUS_SOLUTION"] +".pkl","rb")
        dict_solutions = pickle.load(pickled_dict)

        # Get all the test soltions (every 50 gen) and the corresponding error
        solutions = dict_solutions["test_solutions"]
        solutions_error = dict_solutions["error"]

        # Find the lowest error and find the corresponding solution that achieved that error
        best_sol_ind = np.argmin(solutions_error)
        center_init = solutions[best_sol_ind+1] #+1 since first of solution_testrun is only zeros

        if problem.number_parameters != len(center_init):
            raise ValueError("Number of neurons in initial solution  does not correspond with the initial set number of neurons in the config file")

    elif problem.config["MEAN_SETTING"] == "same for all" or problem.config["MEAN_SETTING"] == "custom":
        center_init = np.array([])
        init_method = problem.config["SAME_FOR_ALL"]

        for name, value in param_model.items():
            number_of_params = len(torch.flatten(value).detach().numpy())

            # If custom, select the init type for that parameter
            if problem.config["MEAN_SETTING"] == "custom":
                init_method = problem.config["CUSTOM"][name]
            print ("Mean initialization of  ", name, " with method ", init_method)

            for iteration in range(number_of_params):

                #Fill the array with the initial parameter
                if init_method == "manual":    init_param =  problem.config[init_method][name][iteration]
                if init_method == "gaussian":  init_param = problem.config[init_method][name]
                if init_method == "range":     init_param = random.uniform(problem.config[init_method][name][0],problem.config[init_method][name][1])
                center_init = np.append(center_init, init_param)

                #Check if extra rule applies to the initialization
                if init_method == "gaus" or init_method == "range":
                    
                    if problem.config["INIT_W1_H2_NEG"]==True and name == "l1.ff.weight" and problem.config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"] == False:
                        if iteration>=number_of_params/2:
                            center_init[-1]=-center_init[-1]
                    
                    if problem.config["INIT_W2_Q2_Q4_NEG"]==True and name == "l2.ff.weight" and problem.config["LAYER_SETTING"]["l2"]["shared_weight_and_bias"] == False:
                        if math.floor(number_of_params/4) < iteration< math.floor(number_of_params/2) or iteration> math.floor(number_of_params*3/4):
                            center_init[-1]=-center_init[-1]
                    
                    if problem.config["INIT_LEAKI_HALF_ZERO"]==True and name == "l1.neuron.leak_i" and problem.config["LAYER_SETTING"]["l1"]["shared_leak_i"] == False:
                        if iteration<number_of_params/2:
                            center_init[-1]=0

    return center_init, std_init

def test_solution(problem, solution):
    # Initialize varibales from problem Class
    input_data, fitness_target = get_dataset(problem.config, None, problem.config["TEST_SIM_TIME"])
    
    #################    Test sequence       ############################
    solution_np = solution.values.detach().numpy() #numpy array of all parameters
    controller = copy.deepcopy(problem.model)
    final_parameters =torchga.model_weights_as_dict(controller, solution_np)
    controller.load_state_dict(final_parameters)

    # Convert the final parameters in the correct shape
    if problem.config["LAYER_SETTING"]["l0"]["enabled"]: controller.l0.init_reshape()
    controller.l1.init_reshape()
    controller.l2.init_reshape()

    if  problem.fitness_mode== 1: #Only controller simlation
        fitness_measured, state_l1_arr, state_l0_arr = run_controller(controller,input_data, save_mode=True)
    elif problem.fitness_mode == 2 or problem.fitness_mode == 3:#Also simulate the dyanmics
        fitness_measured, error, state_l0_arr, state_l1_arr, state_l2_arr = run_controller_dynamics(problem.config,controller,input_data, save_mode=True)

    # Calculate the fitness value
    fitness_value = evaluate_fitness(problem.fitness_func, fitness_measured, fitness_target)

    # Print the parameters of the best solution to the terminal
    for key, value in final_parameters.items():
        print(key, value)

    if problem.fitness_mode == 1:   label_fitness_measured = "SNN output"; label_fitness_target = "PID output"
    if problem.fitness_mode == 2:   label_fitness_measured = "Blimp Height SNN"; label_fitness_target = "Blimp Height Reference"
    if problem.fitness_mode == 3:   label_fitness_measured = "Blimp Height SNN"; label_fitness_target = "Blimp height PID"

    title = "Controller Response"
    time_test = np.arange(0,problem.config["TEST_SIM_TIME"],problem.config["TIME_STEP"])
    if problem.fitness_mode == 2 or problem.fitness_mode == 3:
        title = "Height control of the Blimp"
        plt.plot(time_test, state_l2_arr, linestyle = "--", color = "k", label = "Control output")
    if problem.fitness_mode == 3:
        plt.plot(time_test, torch.flatten(input_data), linestyle = "--", color = "r", label = "Reference input")
    
    plt.plot(time_test, fitness_measured, color = "b", label=label_fitness_measured)
    plt.plot(time_test, fitness_target, color = 'r',label=label_fitness_target)
    plt.title(title)
    plt.grid()
    plt.legend()
    

    if problem.config["WANDB_LOG"] == True:
        wandb.log({title: plt})

    # plt.show()

def create_bounds(problem, model,config):
    bound_config = config["PARAMETER_BOUNDS"]

    lower_bounds = []
    upper_bounds = []
    ### Get the structure and order of the genome
    param_model =torchga.model_weights_as_dict(model, np.ones(problem.number_parameters)) # fill with dummy inputs

    ### Check if there is a lim in the config and otherwise add None to it
    for name, value in param_model.items():
        number_of_params = len(torch.flatten(value).detach().numpy())

        for iteration in range(number_of_params):
            if name in bound_config:

                # Check if parameters is the weights
                if name == "l1.ff.weight" and config["BOUNDS_W1_H2_NEG"]==True and config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"]== False:
                    # Set the last half of the neurons the the negative bound
                    if iteration>=number_of_params/2:
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(0)
                    else:
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(bound_config[name]["high"])

                # Check if parameters is the weights
                elif name == "l2.ff.weight" and config["BOUND_W2_Q2_Q4_NEG"]==True and config["LAYER_SETTING"]["l2"]["shared_weight_and_bias"] == False:
                    # Set the last half of the neurons the the negative bound
                    if math.floor(number_of_params/4) < iteration< math.floor(number_of_params/2) or iteration> math.floor(number_of_params*3/4):
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(0)
                    else:
                        lower_bounds.append(0)
                        upper_bounds.append(bound_config[name]["high"])
                

                # Check if parameters is the leak_i
                elif name == "l1.neuron.leak_i" and config["BOUND_LEAKI_HALF_ZERO"]==True and config["LAYER_SETTING"]["l1"]["shared_leak_i"] == False:
                    # Set the leak_i first half of neurons to a near zero 
                    if iteration<number_of_params/2:
                        lower_bounds.append(0)
                        upper_bounds.append(0.01)
                    else:
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(bound_config[name]["high"])

                else:
                    lower_bounds.append(bound_config[name]["low"])
                    upper_bounds.append(bound_config[name]["high"])
            else:
                lower_bounds.append(None)
                upper_bounds.append(None)

    return (lower_bounds , upper_bounds)

def save_solution(best_solution, problem):
    wandb_mode = problem.config["WANDB_LOG"]
    save_flag = problem.config["SAVE_LAST_SOLUTION"]
    
    if save_flag == True:
        if wandb_mode == True: 
            file_name =  wandb.run.name

        # IF wandb is not logging --> use date and time as file saving
        else:     
            date_time = datetime.fromtimestamp(time.time())  
            file_name = date_time.strftime("%d-%m-%Y_%H-%M-%S")      


        pickle_out = open(problem.prefix + "Results_EA/Blimp/"+ file_name+ ".pkl","wb")
        test_solutions= {"test_solutions":problem.test_solutions, 
                         "error": problem.error_test_solutions, 
                         "step_size": problem.config["SAVE_TEST_SOLUTION_STEPSIZE"], 
                         "generations":problem.config["GENERATIONS"],
                         "best_solution": best_solution.values.detach().numpy(),
                         "datasets": problem.train_datasets,
                        #  "C":problem.C_matrix,
                         "mean": problem.mean,
                         "sigma":problem.sigma,
                         "model_structure": problem.dict_model_structure,
                         "config": problem.config,
                         "all_param_sol": problem.all_param_test_solutions}
        pickle.dump(test_solutions, pickle_out)
        pickle_out.close()



def create_new_training_set():
    # Insert the test dataset every ... times, otherwise choose a random sequence
    if searcher.step_count%problem.config["SAVE_TEST_SOLUTION_STEPSIZE"] == problem.config["SAVE_TEST_SOLUTION_STEPSIZE"]-1 or searcher.step_count==0 or searcher.steps_count ==problem.config["GENERATIONS"]-1:
        problem.dataset=None
        problem.input_data_new,problem.target_data_new = get_dataset(problem.config,problem.dataset,problem.config["TEST_SIM_TIME"])
        problem.manual_dataset_prev = True
        
    elif searcher.step_count%problem.config["DIFFERENT_DATASET_EVERY_GENERATION"]==0 or problem.manual_dataset_prev==True:
        problem.dataset = randint(0,problem.max_number_of_datasets-1)
        problem.input_data_new, problem.target_data_new = get_dataset(problem.config, problem.dataset, problem.config["SIM_TIME"])
        problem.manual_dataset_prev = False

    print("Training dataset = ", problem.dataset)
    problem.train_datasets = np.append(problem.train_datasets, problem.dataset)


    #Check if multiple processes are called, and then change the instance at all actors if necessary
    if problem.actors is not None:
        for i in range(len(problem.actors)):
            actor = problem.actors[i]
            actor.set.remote("input_data",problem.input_data_new)
            actor.set.remote("target_data",problem.target_data_new)
    if problem.actors is None:
        problem.input_data = problem.input_data_new
        problem.target_data = problem.target_data_new
    
def evaluate_manual_dataset():
    save_per_generation = math.ceil(problem.config["GENERATIONS"]/2000)
    if problem.config["ALGORITHM"] == "pycma" and searcher.step_count%save_per_generation ==0: 
        # if problem.C_matrix is None:
        #     problem.C_matrix = searcher._es.C[np.newaxis, ...]
        # else: problem.C_matrix = np.concatenate((problem.C_matrix, searcher._es.C[np.newaxis, ...]), axis=0)

        if problem.mean is None:
            problem.mean = searcher._es.mean[np.newaxis, ...]
        else: problem.mean = np.concatenate((problem.mean, searcher._es.mean[np.newaxis, ...]), axis=0)

        if problem.sigma is None:
            problem.sigma = searcher._es.sigma[np.newaxis, ...]
        else: problem.sigma = np.concatenate((problem.sigma, searcher._es.sigma[np.newaxis, ...]), axis=0)

        if problem.stds is None:
            problem.stds = searcher._es.stds[np.newaxis, ...]
        else: problem.stds = np.concatenate((problem.stds, searcher._es.stds[np.newaxis, ...]), axis=0)
        # problem.mean = np.append(problem.mean,)

    if problem.config["ALGORITHM"] == "cmaes":
        # if problem.C_matrix is None:
        #     problem.C_matrix = searcher.C.detach().numpy()[np.newaxis, ...]
        # else: problem.C_matrix = np.concatenate((problem.C_matrix, searcher.C.detach().numpy()[np.newaxis, ...]), axis=0)

        if problem.mean is None:
            problem.mean = searcher.m.detach().numpy()[np.newaxis, ...]
        else: problem.mean = np.concatenate((problem.mean, searcher.m.detach().numpy()[np.newaxis, ...]), axis=0)

        if problem.sigma is None:
            problem.sigma = searcher.sigma.detach().numpy()[np.newaxis, ...]
        else: problem.sigma = np.concatenate((problem.sigma, searcher.sigma.detach().numpy()[np.newaxis, ...]), axis=0)
        # problem.mean = np.append(problem.mean,)


    #Evaluate every 50 generations
    if searcher.step_count%problem.config["SAVE_TEST_SOLUTION_STEPSIZE"] ==0 or searcher.step_count==1 or searcher.steps_count ==problem.config["GENERATIONS"]:
        best_in_pop = torch.argmin(searcher.population._evdata)
        best_sol_np = searcher.population.values[best_in_pop].detach().numpy()
        problem.error_test_solutions = np.append(problem.error_test_solutions, searcher.population._evdata[best_in_pop])
        problem.test_solutions = np.vstack((problem.test_solutions, best_sol_np))


        controller = copy.deepcopy(problem.model)
        final_parameters =torchga.model_weights_as_dict(controller, best_sol_np)
        controller.load_state_dict(final_parameters)
        
        # Convert the final parameters in the correct shape
        if problem.config["LAYER_SETTING"]["l0"]["enabled"]: controller.l0.init_reshape()
        controller.l1.init_reshape()
        controller.l2.init_reshape()
        all_param_dict = controller.state_dict()
        problem.all_param_test_solutions[str(searcher.step_count)] = all_param_dict

        # Save the results during the session
        best_solution = searcher.status["best"]
        save_solution(best_solution,problem)
        if problem.config["WANDB_LOG"]:
            wandb.config.update({"test_error": np.min(problem.error_test_solutions)},allow_val_change = True)

    if problem.config["GENERATIONS"] >= 100 and searcher.step_count% int(problem.config["GENERATIONS"]/10) == 0:
        best_solution = searcher.status["best"] 
        test_solution(problem,best_solution)

def plot_evolution_parameters(problem):
    generations = np.size(problem.mean,0)
    parameters_mean = problem.mean #shape (generations, parameters)
    parameters_stds = problem.stds

    # Create dummy list of parameters to get the structure of the NN
    _final_parameters =torchga.model_weights_as_dict(problem.model, parameters_mean[0,:])

    ## Initialize dictonary structre{param: [params_gen1, params_gen2 etc..].}
    full_solution_dict_mean = {key:None for key, _ in _final_parameters.items()}
    full_solution_dict_stds = {key:None for key, _ in _final_parameters.items()}



    ### Fill the dictornary
    for param in full_solution_dict_mean:
        for gen in range(parameters_mean.shape[0]):
            gen_parameters_mean =torchga.model_weights_as_dict(problem.model, parameters_mean[gen,:])
            value = gen_parameters_mean[param]
            value = torch.flatten(value).detach().numpy()
            if full_solution_dict_mean[param] is None: #Check is dict is empty 
                full_solution_dict_mean[param] = value
            else:
                full_solution_dict_mean[param] = np.vstack((full_solution_dict_mean[param],value))
            

            gen_parameters_stds =torchga.model_weights_as_dict(problem.model, parameters_stds[gen,:])
            value = gen_parameters_stds[param]
            value = torch.flatten(value).detach().numpy()
            if full_solution_dict_stds[param] is None: #Check is dict is empty 
                full_solution_dict_stds[param] = value
            else:
                full_solution_dict_stds[param] = np.vstack((full_solution_dict_stds[param],value))



    ### Plot the different diagrams
    gen_arr = np.arange(0,generations)
    for param in full_solution_dict_mean:
        # Do not save the recurrent and l1 if siz is larger than the number of neurons, to save time during plotting
        if len(full_solution_dict_mean[param][0,:]) > problem.config["NEURONS"]:
            continue
        plt.figure()
        for num_param in range(len(full_solution_dict_mean[param][0,:])):
            param_per_gen = full_solution_dict_mean[param][:,num_param]
            param_per_gen = param_per_gen.flatten()
            plt.plot(gen_arr,param_per_gen,label=str(num_param))
        plt.title("Mean of " + str(param))
        plt.xlabel("Generations [-]")
        # plt.xticks(gen_arr)
        plt.legend()
        plt.grid()
        if problem.config["WANDB_LOG"] == True:
            wandb.log({"MEAN of " + str(param): plt})
             
             
        plt.figure()
        for num_param in range(len(full_solution_dict_stds[param][0,:])):
            param_per_gen = full_solution_dict_stds[param][:,num_param]
            param_per_gen = param_per_gen.flatten()
            plt.plot(gen_arr,param_per_gen,label=str(num_param))
        plt.title("STDS of " + str(param))
        plt.xlabel("Generations [-]")
        # plt.xticks(gen_arr)
        plt.legend()
        plt.grid()
        if problem.config["WANDB_LOG"] == True:
            wandb.log({"STDS of " + str(param): plt})


    plt.show(block=problem.config["SHOW_PLOTS"])
                    
    return 

def plot_stepsize(problem):
    sigma = problem.sigma
    gen_arr = np.arange(0, len(sigma))
    plt.figure()
    plt.title("Stepsize over generations")
    plt.xlabel("Generations [-]")
    plt.ylabel("Stepsize [-]")
    plt.plot(gen_arr,sigma)
    plt.grid()
    if problem.config["WANDB_LOG"] == True:
        wandb.log({"Stepsize over generations": plt})

def get_dataset(config, dataset_num, sim_time):
    if platform.system() == "Linux":
        # Check if running on pc or delftblue
        if os.path.isdir("/home/tim"):
            prefix = "/home/tim/SNN_Workspace/"
        else: prefix = "/scratch/timburgers/SNN_Workspace/"


    if platform.system() == "Windows":
        prefix = ""

    time_step = config["TIME_STEP"]

    # Either use one of the standard datasets
    if type(dataset_num) == int: 
        file = "/dataset_"+ str(dataset_num)
        if config["START_DATASETS_IN_MIDDLE"] == True:
            start_in_middle = int(15*(1/time_step))
        else: start_in_middle = 1
    # Or the manual created one
    elif dataset_num ==None: 
        file = "/" + config["TEST_DATA_FILE"]
        start_in_middle=1
    # Or the step functions (longer time inbetween)
    elif dataset_num == "step":
        file = "/step_dataset"
        start_in_middle=1
    
    # Select the correct input and target datasets, based on the "TARGET_FITNESS" in the config
    #column =   0)Z   1)Z_ref   2)Error   3)Kp*error    4)Kd*error    5)PD_output
    if config["ALTERNATIVE_INPUT_COLUMN"] != None and config["ALTERNATIVE_TARGET_COLUMN"] != None: input_col = config["ALTERNATIVE_INPUT_COLUMN"]; target_col = config["ALTERNATIVE_TARGET_COLUMN"]
    elif config["TARGET_FITNESS"] == 1:     input_col = [2]; target_col = [5]
    elif config["TARGET_FITNESS"] == 2:     input_col = [1]; target_col = [1]
    elif config["TARGET_FITNESS"] == 3:     input_col = [1]; target_col = [0]

    input_data = pd.read_csv(prefix + config["DATASET_DIR"]+ file + ".csv", usecols=input_col, header=None, skiprows=start_in_middle, nrows=sim_time*(1/time_step))
    input_data = torch.tensor(input_data.values).float().unsqueeze(0).unsqueeze(2) 	# convert from pandas df to torch tensor and floats + shape from (seq_len ,features) to (1, seq, feature)


    target_data = pd.read_csv(prefix + config["DATASET_DIR"] + file + ".csv", usecols=target_col, header=None,  skiprows=start_in_middle, nrows=sim_time*(1/time_step))
    target_data = torch.tensor(target_data.values).float()
    target_data = target_data[:,0]



    return input_data, target_data

def changes_names_in_table_wandb(config):
    config_cop = copy.deepcopy(config)
    lay_set = config_cop["LAYER_SETTING"]
    for layer_name, layer_settings in lay_set.items():
        for set_name, set_value in layer_settings.items():
            if set_value == False or set_value == None:
                config_cop["LAYER_SETTING"][layer_name][set_name] = "-"
            if set_value == True:
                config_cop["LAYER_SETTING"][layer_name][set_name] = "X"

    wandb.config.update({   "0) En":   config_cop["LAYER_SETTING"]["l0"]["enabled"],
                            "0) N":    config_cop["LAYER_SETTING"]["l0"]["neurons"],
                            "0) b":    config_cop["LAYER_SETTING"]["l0"]["bias"],
                            "0) wbS":  config_cop["LAYER_SETTING"]["l0"]["shared_weight_and_bias"],
                            "0) iS":   config_cop["LAYER_SETTING"]["l0"]["shared_leak_i"],

                            "1) N":    config_cop["NEURONS"],
                            "1) R":    config_cop["LAYER_SETTING"]["l1"]["recurrent"],
                            "1) R":    config_cop["LAYER_SETTING"]["l1"]["recurrent"],
                            "1) A":    config_cop["LAYER_SETTING"]["l1"]["adaptive"],
                            "1) AI":   config_cop["LAYER_SETTING"]["l1"]["adapt_thres_input_spikes"],
                            "1) A2x2": config_cop["LAYER_SETTING"]["l1"]["adapt_2x2_connection"],
                            "1) AS-add":   config_cop["LAYER_SETTING"]["l1"]["adapt_share_add_t"],
                            "1) AS-b&l":   config_cop["LAYER_SETTING"]["l1"]["adapt_share_baseleak_t"],
                            "1) b":    config_cop["LAYER_SETTING"]["l1"]["bias"],
                            "1) wD":   config_cop["LAYER_SETTING"]["l1"]["w_diagonal"],
                            "1) wD2x2":config_cop["LAYER_SETTING"]["l1"]["w_diagonal_2x2"],
                            "1) wbS":  config_cop["LAYER_SETTING"]["l1"]["shared_weight_and_bias"],
                            "1) w2x2S-crs":  config_cop["LAYER_SETTING"]["l1"]["shared_2x2_weight_cross"],
                            "1) iS":   config_cop["LAYER_SETTING"]["l1"]["shared_leak_i"],

                            "2) ComL": config_cop["LAYER_SETTING"]["l2"]["complementary_leak"],
                            "2) wbS":  config_cop["LAYER_SETTING"]["l2"]["shared_weight_and_bias"]})

if __name__ == "__main__":
    config_folder = "configs/"
    opts, args = getopt.getopt(sys.argv[1:], "c:",["config="])
    for opt,arg in opts:
        if opt in ("-c", "--config"):
            config_path = config_folder + arg

    # If not optional command are provided, use default config file
    if len(opts) ==0:
        config_path = config_folder + "config_LIF_DEFAULT"
    print("Config file used = " +  config_path)
    
    problem = LIF_EA_evotorch(config_path)
    center_init, std_init = init_conditions(problem)
    
    if problem.config["ALGORITHM"]=="cmaes": searcher = CMAES(problem, stdev_init=1, center_init=center_init, limit_C_decomposition=False, popsize=problem.config["INDIVIDUALS"])
    if problem.config["ALGORITHM"]=="pycma": searcher = PyCMAES(problem,stdev_init=0.4, stdev_max=1, center_init=center_init,  popsize=problem.config["INDIVIDUALS"], cma_options={"CMA_stds":std_init}) # Porblem with bound, the mean drifts off, far off the limits ("fixed" with a repair function)

    ### Insert a new dataset every generation
    if problem.config["ANTI_OVERFITTING"]:
        searcher.before_step_hook.append(create_new_training_set)
    if problem.config["SAVE_TEST_SOLUTION_STEPSIZE"] is not None:
        searcher.after_step_hook.append(evaluate_manual_dataset)

    if problem.config["WANDB_LOG"] == True:
        _ = WandbLogger(searcher, project = "blimp_real_hz", config=problem.config)
        changes_names_in_table_wandb(problem.config)
        wandb.config.update({"OS": platform.system(),
                             "Fit_Fn": problem.config["FITNESS_FUNCTION"]})
        if problem.config["MEAN_SETTING"]== "same for all": wandb.config.update({"MEAN_SETTING": problem.config["SAME_FOR_ALL"]},allow_val_change = True)
        if problem.config["MEAN_SETTING"]== "previous": wandb.config.update({"MEAN_SETTING": problem.config["PREVIOUS_SOLUTION"]},allow_val_change = True)

        number_first_wandb_name()
    # logger = StdOutLogger(searcher)


    searcher.run(problem.config["GENERATIONS"])
    best_solution = searcher.status["best"]
    save_solution(best_solution,problem)
    test_solution(problem,best_solution)
    plot_evolution_parameters(problem)
    plot_stepsize(problem)
