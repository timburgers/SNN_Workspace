
import torch
import pygad.torchga as torchga
from SNN_LIF_LI_init import LIF_SNN
from wandb_log_functions import number_first_wandb_name
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import os
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
            self.prefix = "/scratch/timburgers/SNN_Workspace/"

        if platform.system() == "Windows":
            self.prefix = ""

        ### Read config file
        with open(self.prefix + config_path + ".yaml","r") as f:
            self.config = yaml.safe_load(f)
        
        # Select model
        self.model = LIF_SNN(None, self.config["NEURONS"], self.config["LAYER_SETTING"])
        self.number_parameters = 0

        # Log the structure of the model that is used
        self.dict_model_structure = dict()
        for name, parameter in self.model.named_parameters():
            self.number_parameters = self.number_parameters + torch.flatten(parameter).shape[0]
            self.dict_model_structure[name] = torch.flatten(parameter).shape[0]

        
        self.input_data, self.target_data = get_dataset(self.config, self.config["DATASET_NUMBER"], self.config["SIM_TIME"])
        self.mse = torch.nn.MSELoss()
        self.pearson = PearsonCorrCoef()
        self.bounds = create_bounds(self, self.model, self.config)
        self.prefix = ""
        self.test_solutions= np.zeros(self.number_parameters)
        self.error_test_solutions= np.array([])
        self.train_datasets=np.array([])
        self.manual_dataset_prev = False
        self.fitness_mode = self.config["TARGET_FITNESS"]

        self.datatset = None
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
        #### Initialize neuron states (u, v, s) 
        snn_states = torch.zeros(3, 1, self.model.neurons) # frist dim in order [current,mempot,spike]
        LI_state = torch.zeros(1,1)

        solution_np = solution.values.detach().numpy() #numpy array of all parameters
        controller = copy.deepcopy(self.model)
        final_parameters =torchga.model_weights_as_dict(controller, solution_np)
        controller.load_state_dict(final_parameters)

        if  self.fitness_mode== 1: #Only controller simlation
            fitness_measured = run_controller(self,controller,self.input_data, snn_states,LI_state, save_mode = False)
        elif self.fitness_mode == 2 or self.fitness_mode == 3:#Also simulate the dyanmics
            fitness_measured = run_controller_dynamics(self,controller,self.input_data, snn_states,LI_state, save_mode = False)

        # Calculate the fitness value
        fitness_value = evaluate_fitness(self, fitness_measured, self.target_data)

        print(fitness_value)
        solution.set_evals(fitness_value)


def run_controller(self,controller,input, snn_states, LI_state,save_mode):
    control_output = np.array([])
    if save_mode == False:
        for t in range(input.shape[1]):
            error = input[:,t,:]
            snn_spikes, snn_states, LI_state = controller(error, snn_states, LI_state)

            # Append states to array
            control_output = np.append(control_output,LI_state.detach().numpy())
        return control_output
    
    elif save_mode == True:
        for t in range(input.shape[1]):
            error = input[:,t,:]
            snn_spikes, snn_states, LI_state = controller(error, snn_states, LI_state)

            # Append states to array
            if t==0: control_state = snn_states.detach().numpy()[np.newaxis,...]
            else: control_state = np.concatenate((control_state, snn_states.detach().numpy()[np.newaxis, ...]))
            control_output = np.append(control_output,LI_state.detach().numpy())
        return control_output, control_state


def run_controller_dynamics(self,controller,input, snn_states, LI_state, save_mode):
    dyn_system = Blimp(self.config)
    sys_output = np.array([0])
    
    if save_mode == False:
        for t in range(input.shape[1]):
            ref = input[:,t,:]
            error = ref - sys_output[-1]
            snn_spikes, snn_states, LI_state = controller(error, snn_states, LI_state)
            dyn_system.sim_dynamics(LI_state.detach().numpy())

            #Append states to array
            sys_output = np.append(sys_output,dyn_system.get_z())
        return sys_output[1:] # Skip the first 0 height input

    if save_mode==True:
        control_input = np.array([])
        control_output = np.array([])

        for t in range(input.shape[1]):
            ref = input[:,t,:]
            error = ref - sys_output[-1]
            snn_spikes, snn_states, LI_state = controller(error, snn_states, LI_state)
            dyn_system.sim_dynamics(LI_state.detach().numpy())

            #Append states to array
            if t==0: control_state = snn_states.detach().numpy()[np.newaxis,...]
            else: control_state = np.concatenate((control_state, snn_states.detach().numpy()[np.newaxis, ...]))
            control_input = np.append(control_input, error.detach().numpy())
            control_output = np.append(control_output, LI_state.detach().numpy())
            sys_output = np.append(sys_output,dyn_system.get_z())
        return sys_output[1:], control_input, control_state, control_output # Skip the first 0 height input


def evaluate_fitness(self, fitness_measured, fitness_target):
    #Evaluate fitness using MSE and additionally pearson if there should be a linear correlation between target and output
    fitness_target = torch.flatten(fitness_target)
    fitness_measured = torch.from_numpy(fitness_measured)
    fitness_value = self.mse(fitness_measured,fitness_target)
    if self.fitness_mode == 1 or self.fitness_mode == 3:
        fitness_value += (1-self.pearson(fitness_measured,fitness_target)) #pearson of 1 means linear correlation
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
    input_data, fitness_target = get_dataset(problem.config, None, problem.config["SIM_TIME"])
    
    #################    Test sequence       ############################
    snn_states = torch.zeros(3, 1, problem.model.neurons) # frist dim in order [current,mempot,spike]
    LI_state = torch.zeros(1,1)

    solution_np = solution.values.detach().numpy() #numpy array of all parameters
    controller = copy.deepcopy(problem.model)
    final_parameters =torchga.model_weights_as_dict(controller, solution_np)
    controller.load_state_dict(final_parameters)

    if  problem.fitness_mode== 1: #Only controller simlation
        fitness_measured, control_state = run_controller(problem,controller,input_data, snn_states,LI_state, save_mode=True)
    elif problem.fitness_mode == 2 or problem.fitness_mode == 3:#Also simulate the dyanmics
        fitness_measured, control_input, control_state, control_output = run_controller_dynamics(problem,controller,input_data, snn_states,LI_state, save_mode=True)

    # Calculate the fitness value
    fitness_value = evaluate_fitness(problem, fitness_measured, fitness_target)

    # Print the parameters of the best solution to the terminal
    for key, value in final_parameters.items():
        print(key, value)

    if problem.fitness_mode == 1:   label_fitness_measured = "SNN output"; label_fitness_target = "PID output"
    if problem.fitness_mode == 2:   label_fitness_measured = "Blimp Height SNN"; label_fitness_target = "Blimp Height Reference"
    if problem.fitness_mode == 3:   label_fitness_measured = "Blimp Height SNN"; label_fitness_target = "Blimp height PID"

    title = "Controller Response"
    time_test = np.arange(0,problem.config["SIM_TIME"],problem.config["TIME_STEP"])
    if problem.fitness_mode == 2 or problem.fitness_mode == 3:
        title = "Height control of the Blimp"
        plt.plot(time_test, control_output, linestyle = "--", color = "k", label = "Control output")
    if problem.fitness_mode == 3:
        plt.plot(time_test, torch.flatten(input_data), linestyle = "--", color = "r", label = "Reference input")
    
    plt.plot(time_test, fitness_measured, color = "b", label=label_fitness_measured)
    plt.plot(time_test, fitness_target, color = 'r',label=label_fitness_target)
    plt.title(title)
    plt.grid()
    plt.legend()
    

    if problem.config["WANDB_LOG"] == True:
        wandb.log({title: plt})

    plt.show()

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
                         "config": problem.config}
        pickle.dump(test_solutions, pickle_out)
        pickle_out.close()

def create_new_training_set():
    # Insert the test dataset every ... times, otherwise choose a random sequence
    if searcher.step_count%problem.config["SAVE_TEST_SOLUTION_STEPSIZE"] == problem.config["SAVE_TEST_SOLUTION_STEPSIZE"]-1 or searcher.step_count==0 or searcher.steps_count ==problem.config["GENERATIONS"]-1:
        problem.dataset=None
        problem.input_data_new,problem.target_data_new = get_dataset(problem.config,problem.dataset,20)
        problem.manual_dataset_prev = True
        
    elif searcher.step_count%problem.config["DIFFERENT_DATASET_EVERY_GENERATION"]==0 or problem.manual_dataset_prev==True:
        problem.dataset = randint(0,499)
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
        best_sol = searcher.population.values[best_in_pop].detach().numpy()
        problem.error_test_solutions = np.append(problem.error_test_solutions, searcher.population._evdata[best_in_pop])
        problem.test_solutions = np.vstack((problem.test_solutions, best_sol))

        # Save the results during the session
        best_solution = searcher.status["best"]
        save_solution(best_solution,problem)

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
        for gen in range(generations):
            gen_parameters_mean =torchga.model_weights_as_dict(problem.model, parameters_mean[gen,:])
            for name, value in gen_parameters_mean.items():
                if param == name:
                    value = torch.flatten(value).detach().numpy()
                    if full_solution_dict_mean[param] is None: #Check is dict is empty 
                        full_solution_dict_mean[param] = value
                    else:
                        full_solution_dict_mean[param] = np.vstack((full_solution_dict_mean[param],value))
                    break

            gen_parameters_stds =torchga.model_weights_as_dict(problem.model, parameters_stds[gen,:])
            for name, value in gen_parameters_stds.items():
                if param == name:
                    value = torch.flatten(value).detach().numpy()
                    if full_solution_dict_stds[param] is None: #Check is dict is empty 
                        full_solution_dict_stds[param] = value
                    else:
                        full_solution_dict_stds[param] = np.vstack((full_solution_dict_stds[param],value))
                    break


    ### Plot the different diagrams
    gen_arr = np.arange(0,generations)
    for param in full_solution_dict_mean:
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
        prefix = "/scratch/timburgers/SNN_Workspace/"

    if platform.system() == "Windows":
        prefix = ""

    time_step = config["TIME_STEP"]

    # Either use one of the standard datasets
    if dataset_num != None:
        file = "/dataset_"+ str(dataset_num)
        if config["START_DATASETS_IN_MIDDLE"] == True:
            start_in_middle = 15*(1/time_step)
        else: start_in_middle = 1
    # Or the manual created one
    else: 
        file = "/" + config["TEST_DATA_FILE"]
        start_in_middle=1
    
    # Select the correct input and target datasets, based on the "TARGET_FITNESS" in the config
    #column =   0)Z   1)Z_ref   2)Error   3)Kp*error    4)Kd*error    5)PD_output
    if config["TARGET_FITNESS"] == 1:       input_col = [2]; target_col = [5]
    elif config["TARGET_FITNESS"] == 2:     input_col = [1]; target_col = [1]
    elif config["TARGET_FITNESS"] == 3:     input_col = [1]; target_col = [0]

    input_data = pd.read_csv(prefix + config["DATASET_DIR"]+ file + ".csv", usecols=input_col, header=None, skiprows=start_in_middle, nrows=sim_time*(1/time_step))
    input_data = torch.tensor(input_data.values).float().unsqueeze(0).unsqueeze(2) 	# convert from pandas df to torch tensor and floats + shape from (seq_len ,features) to (1, seq, feature)


    target_data = pd.read_csv(prefix + config["DATASET_DIR"] + file + ".csv", usecols=target_col, header=None,  skiprows=start_in_middle, nrows=sim_time*(1/time_step))
    target_data = torch.tensor(target_data.values).float()
    target_data = target_data[:,0]



    return input_data, target_data


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
        _ = WandbLogger(searcher, project = "SNN_BLIMP", config=problem.config)
        wandb.config.update({"OS": platform.system()})
        number_first_wandb_name()
    # logger = StdOutLogger(searcher)


    searcher.run(problem.config["GENERATIONS"])

    best_solution = searcher.status["best"]
    save_solution(best_solution,problem)
    test_solution(problem,best_solution)
    plot_evolution_parameters(problem)
    plot_stepsize(problem)
