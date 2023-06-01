
import torch
import pygad.torchga as torchga
from SNN_LIF_LI_init import LIF_SNN
from wandb_log_functions import number_first_wandb_name
from IZH.Izh_LI_EA_PYGAD import get_dataset
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

import warnings
warnings.filterwarnings("ignore")
ray.init(log_to_driver=False, include_dashboard=False)

class LIF_EA_evotorch(Problem):
    def __init__(self, config):
        if platform.system() == "Linux":
            self.prefix = "/scratch/timburgers/SNN_Workspace/"

        if platform.system() == "Windows":
            self.prefix = ""

        ### Read config file
        with open(self.prefix + config + ".yaml","r") as f:
            self.config = yaml.safe_load(f)

        self.model = LIF_SNN(None, "cpu", self.config)
        self.number_parameters = self.model.neurons**2+ self.model.neurons*6+1
        self.input_data, self.target_data = get_dataset(self.config, self.config["DATASET_NUMBER"], self.config["SIM_TIME"])
        self.mse = torch.nn.MSELoss()
        self.pearson = PearsonCorrCoef()
        self.bounds = create_bounds(self, self.model, self.config)
        self.prefix = ""
        self.test_solutions= np.zeros(self.number_parameters)
        self.error_test_solutions= np.array([])
        self.train_datasets=np.array([])
        self.manual_dataset_prev = False

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
        solution_np = solution.values.detach().numpy()


        if self.config["FITNESS_INCLUDE_DYANMICS"] == False:
            _, _, predictions = torchga.predict(self.model,
                                                solution_np,
                                                self.input_data,
                                                snn_states, #(u, v, s) 
                                                LI_state) #data in form: input, state_snn, state LI

            predictions = predictions[:,0,0]
            pearson_loss = 1-self.pearson(predictions, self.target_data)
            mse_loss = self.mse(predictions, self.target_data)
            solution_fitness = (mse_loss + pearson_loss).detach().numpy()
        
        if self.config["FITNESS_INCLUDE_DYANMICS"] == True:
            model_weights_dict = torchga.model_weights_as_dict(self.model,solution_np)
            _model = copy.deepcopy(self.model)
            _model.load_state_dict(model_weights_dict)
            dyn_system = Blimp(self.config)
            height = torch.tensor([0])
            input_data = self.input_data[..., np.newaxis]


            for t in range(input_data.shape[1]):
                ref = input_data[:,t,:,:]
                error = ref - height[-1]
                _, snn_states, LI_state = _model(error, snn_states, LI_state)
                snn_states = snn_states[0,:,:,:]    # get rid of the additional list where the items are inserted in forward pass
                LI_state = LI_state[0,:,:]          # get rid of the additional list where the items are inserted in forward pass
                dyn_system.sim_dynamics(LI_state.detach().numpy())
                height = np.append(height, dyn_system.get_z())

            mse_loss =(np.square(height[1:] - self.target_data.detach().numpy())).mean() # Skip the first 0 height input
            solution_fitness = mse_loss

        
        print(solution_fitness)
        solution.set_evals(solution_fitness)
    

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
                    
                    if problem.config["INIT_W1_H2_NEG"]==True and name == "l1.ff.weight":
                        if iteration>=number_of_params/2:
                            center_init[-1]=-center_init[-1]
                    
                    if problem.config["INIT_W2_Q2_Q4_NEG"]==True and name == "l2.ff.weight":
                        if math.floor(number_of_params/4) < iteration< math.floor(number_of_params/2) or iteration> math.floor(number_of_params*3/4):
                            center_init[-1]=-center_init[-1]
                    
                    if problem.config["INIT_LEAKI_HALF_ZERO"]==True and name == "l1.neuron.leak_i":
                        if iteration<number_of_params/2:
                            center_init[-1]=0

    return center_init, std_init

def test_solution(problem, solution):
    # Initialize varibales from problem Class
    model = problem.model
    test_input_data, test_target_data = get_dataset(problem.config, None, 20)
    solution = solution.values.detach().numpy()

    ### Print the final parameters to the terminal
    final_parameters =torchga.model_weights_as_dict(model, solution)
    for key, value in final_parameters.items():
        print(key, value)


    #################    Test sequence       ############################
    #### Initialize neuron states (u, v, s) 
    snn_states = torch.zeros(3, 1, model.neurons)
    LI_state = torch.zeros(1,1)
    

    if problem.config["FITNESS_INCLUDE_DYANMICS"] == False:
        # Make predictions based on the best solution.
        l1_spikes, l1_state, control_output = torchga.predict(model,
                                            solution,
                                            test_input_data,
                                            snn_states,
                                            LI_state)

        actual_data = control_output[:,0,0]
        target_data = test_target_data

        mse_pearson = problem.mse(actual_data, target_data) + (1-problem.pearson(actual_data, target_data))

        actual_data = actual_data.detach().numpy()
        target_data = target_data.detach().numpy()
        print("MSE + Peasrons loss = ", mse_pearson.detach().numpy())

        label_actual_data = "Output of controller"
        title = "Controller output and reference"

    if problem.config["FITNESS_INCLUDE_DYANMICS"] == True:
        model_weights_dict = torchga.model_weights_as_dict(problem.model,solution)
        _model = copy.deepcopy(problem.model)
        _model.load_state_dict(model_weights_dict)
        dyn_system = Blimp(problem.config)
        sys_output = torch.tensor([0])
        control_output = np.array([])
        input_data = test_input_data[..., np.newaxis]


        for t in range(input_data.shape[1]):
            ref = input_data[:,t,:,:]
            error = ref - sys_output[-1]
            _, snn_states, LI_state = _model(error, snn_states, LI_state)
            snn_states = snn_states[0,:,:,:]    # get rid of the additional list where the items are inserted in forward pass
            LI_state = LI_state[0,:,:]         # get rid of the additional list where the items are inserted in forward pass
            dyn_system.sim_dynamics(LI_state.detach().numpy())
            control_output = np.append(control_output, LI_state.detach().numpy())
            sys_output = np.append(sys_output, dyn_system.get_z())
        
        actual_data = sys_output[1:]
        target_data = test_target_data.detach().numpy()

        mse_loss =(np.square(actual_data -target_data)).mean() # Skip the first 0 height input
        print("MSE = ", mse_loss)

        label_actual_data = "Height of blimp"
        title = "Height control of the Blimp"
        
        time_test = np.arange(0,np.size(actual_data)*problem.config["TIME_STEP"],problem.config["TIME_STEP"])
        plt.plot(time_test, control_output, linestyle = "--", color = "k" )


    time_test = np.arange(0,np.size(actual_data)*problem.config["TIME_STEP"],problem.config["TIME_STEP"])
    plt.plot(time_test, actual_data, color = "b", label=label_actual_data)
    plt.plot(time_test,test_target_data, color = 'r',label="Target")
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
                if name == "l1.ff.weight" and config["BOUNDS_W1_H2_NEG"]==True:
                    # Set the last half of the neurons the the negative bound
                    if iteration>=number_of_params/2:
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(0)
                    else:
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(bound_config[name]["high"])

                # Check if parameters is the weights
                elif name == "l2.ff.weight" and config["BOUND_W2_Q2_Q4_NEG"]==True:
                    # Set the last half of the neurons the the negative bound
                    if math.floor(number_of_params/4) < iteration< math.floor(number_of_params/2) or iteration> math.floor(number_of_params*3/4):
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(0)
                    else:
                        lower_bounds.append(0)
                        upper_bounds.append(bound_config[name]["high"])
                

                # Check if parameters is the leak_i
                elif name == "l1.neuron.leak_i" and config["BOUND_LEAKI_HALF_ZERO"]==True:
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


        pickle_out = open(problem.prefix + "Results_EA/LIF/Evotorch/"+ file_name+ ".pkl","wb")
        test_solutions= {"test_solutions":problem.test_solutions, 
                         "error": problem.error_test_solutions, 
                         "step_size": problem.config["SAVE_TEST_SOLUTION_STEPSIZE"], 
                         "generations":problem.config["GENERATIONS"],
                         "best_solution": best_solution.values.detach().numpy(),
                         "datasets": problem.train_datasets,
                        #  "C":problem.C_matrix,
                         "mean": problem.mean,
                         "sigma":problem.sigma}
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

if __name__ == "__main__":
    config_folder = "configs/"
    opts, args = getopt.getopt(sys.argv[1:], "c:",["config="])
    for opt,arg in opts:
        if opt in ("-c", "--config"):
            config = config_folder + arg

    # If not optional command are provided, use default config file
    if len(opts) ==0:
        config = config_folder + "config_LIF_DEFAULT"
    print("Config file used = " +  config)
    
    problem = LIF_EA_evotorch(config)
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
