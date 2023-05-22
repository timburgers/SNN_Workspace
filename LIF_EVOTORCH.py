
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

from evotorch import Problem
from evotorch.algorithms import CMAES, PyCMAES
from evotorch.logging import StdOutLogger,WandbLogger
import ray
import wandb
from datetime import datetime
import time
import platform
import math

import warnings
warnings.filterwarnings("ignore")
ray.init(log_to_driver=False, include_dashboard=False)

class LIF_EA_evotorch(Problem):
    def __init__(self):
        if platform.system() == "Linux":
            self.prefix = "/scratch/timburgers/SNN_Workspace/"

        if platform.system() == "Windows":
            self.prefix = ""

        ### Read config file
        with open(self.prefix + "config_LIF_EVOTORCH.yaml","r") as f:
            self.config = yaml.safe_load(f)

        self.model = LIF_SNN(None, "cpu", self.config)
        self.number_parameters = self.model.neurons**2+ self.model.neurons*6+1
        self.input_data, self.target_data = get_dataset(self.config, self.config["DATASET_NUMBER"], self.config["SIM_TIME"])
        self.loss_function = torch.nn.MSELoss()
        self.bounds = create_bounds(self, self.model, self.config)
        self.prefix = ""
        self.test_solutions= np.zeros(self.number_parameters)
        self.error_test_solutions= np.array([])
        self.train_datasets=np.array([])
        self.manual_dataset_prev = False

        self.datatset = None
        self.input_data_new = None
        self.target_data_new = None

        self.C_matrix = None
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
        

        _, _, predictions = torchga.predict(self.model,
                                            solution_np,
                                            self.input_data,
                                            snn_states, #(u, v, s) 
                                            LI_state) #data in form: input, state_snn, state LI

        predictions = predictions[:,0,0].detach().numpy()
        target_data = self.target_data.detach().numpy()
        print("max target data = ", np.max(target_data))
        solution_fitness = (np.square(predictions - target_data)).mean(axis=None)
        # solution_fitness = self.loss_function(predictions, self.target_data).detach().numpy()
        print(solution_fitness)
        solution.set_evals(solution_fitness)
    

# Get the intiail position of the center and of the step size
def init_conditions(problem):
    init_config = problem.config["INITIAL_PARAMS_RANDOM"]
    bounds_config= problem.config["PARAMETER_BOUNDS"] #The bounds are usedto scale the initial step size
    model = problem.model
    names_center_init = np.array([])
    center_init = np.array([])
    std_init=np.array([])
    manual_w1_ind = 0
    manual_w2_ind = 0
    manual_leaki_ind = 0


    ### Get the structure and order of the genome
    param_model =torchga.model_weights_as_dict(model, np.ones(problem.number_parameters))

    # Check if both initialization parameters are enabled
    if isinstance(problem.config["MANUAL_W1"],list) and problem.config["INIT_W1_SMALL_WITH_MORE_LEAK"]==True:
        raise Exception("Can not have manual weitghs and small weights for small leak in config")
    if isinstance(problem.config["MANUAL_W2"],list) and problem.config["INIT_W2_HALF_NEG"]==True:
        raise Exception("Can not have manual weitghs and half negative weights in config")
    if isinstance(problem.config["MANUAL_leaki"],list) and problem.config["INIT_LEAKI_HALF_ZERO"]==True:
        raise Exception("Can not have manual leak_i and half leak zero in config")
    
    #Check if list and then is size is same as the number of neurons
    if isinstance(problem.config["MANUAL_W1"],list) and isinstance(problem.config["MANUAL_W2"],list) and isinstance(problem.config["MANUAL_leaki"],list):
        if len(problem.config["MANUAL_W1"]) != problem.config["NEURONS"] or len(problem.config["MANUAL_W2"]) != problem.config["NEURONS"] or len(problem.config["MANUAL_leaki"]) != problem.config["NEURONS"]:
            raise Exception("Manual init of weight or leaks not same size as the number of neurons")

    ### Check if there is a lim in the config and otherwise add None to it
    for name, value in param_model.items():
        number_of_params = len(torch.flatten(value).detach().numpy())
        for iteration in range(number_of_params):
            
            # Fill in initial condition of mu
            if name in init_config:
                center_init = np.append(center_init,init_config[name])

                ### Set Weight 1
                if name =="l1.ff.weight":  
                    
                    # Manually set weight 1
                    if isinstance(problem.config["MANUAL_W1"],list):
                        manual_w1 = problem.config["MANUAL_W1"]
                        center_init[-1]=manual_w1[manual_w1_ind]
                        manual_w1_ind +=1


                    # Set the last half (with small leak), with a smaller weights
                    if problem.config["INIT_W1_SMALL_WITH_MORE_LEAK"]==True:
                        if iteration<number_of_params/2:
                            center_init[-1]=center_init[-1]/10
                
                
                ### Set bias 1
                if name =="l1.ff.bias":  

                    # Set the last half (with small leak), with a smaller weights
                    if problem.config["INIT_BIAS1_HALF_NEG"]==True:
                        if iteration>=number_of_params/2:
                            center_init[-1]=-center_init[-1]


                ### Set Weight 2
                if name =="l2.ff.weight":

                    # Manually set weight 2
                    if isinstance(problem.config["MANUAL_W2"],list):
                        manual_w2 = problem.config["MANUAL_W2"]
                        center_init[-1]=manual_w2[manual_w2_ind]
                        manual_w2_ind +=1


                    if problem.config["INIT_W2_HALF_NEG"]==True:
                        if iteration>=number_of_params/2:
                            center_init[-1]=-center_init[-1]
                

                ### Set leak_i
                if  name.split(".")[-1]=="leak_i":

                    # Manually set leak_i
                    if isinstance(problem.config["MANUAL_leaki"],list):
                        manual_leaki = problem.config["MANUAL_leaki"]
                        center_init[-1]=manual_leaki[manual_leaki_ind]
                        manual_leaki_ind +=1
                
                    if problem.config["INIT_LEAKI_HALF_ZERO"]==True:
                        if iteration<number_of_params/2:
                            center_init[-1]=0


            else:
                center_init = np.append(center_init,init_config[name])

            # Create list with names of paramters
            names_center_init = np.append(names_center_init,name)

            # Fill in initial condition of stepsize (std)
            initial_step_size = (bounds_config[name]["high"]-bounds_config[name]["low"])*problem.config["PERCENT_INTIIAL_STEPSIZE"]
            std_init = np.append(std_init,initial_step_size)

    return center_init, std_init

def test_solution(problem, solution):
    # Initialize varibales from problem Class
    model = problem.model
    test_input_data, test_target_data = get_dataset(problem.config, None, 13)
    solution = solution.values.detach().numpy()

    ### Print the final parameters to the terminal
    final_parameters =torchga.model_weights_as_dict(model, solution)
    for key, value in final_parameters.items():
        print(key, value)


    #################    Test sequence       ############################
    #### Initialize neuron states (u, v, s) 
    snn_states = torch.zeros(3, 1, model.neurons)
    LI_state = torch.zeros(1,1)

    # Make predictions based on the best solution.
    l1_spikes, l1_state, predictions = torchga.predict(model,
                                        solution,
                                        test_input_data,
                                        snn_states,
                                        LI_state)
    predictions = predictions[:,0,0]
    
    spike_train = l1_state[:,2,0,:].detach().numpy()
    # create_wandb_summary_table_EA(run, spike_train, config, final_parameters)

    abs_error = problem.loss_function(predictions, test_target_data)

    print("\n Predictions : \n", predictions.detach().numpy())
    print("Absolute Error : ", abs_error.detach().numpy())


    predictions = predictions.detach().numpy()
    test_target_data = test_target_data.detach().numpy()
    time_test = np.arange(0,np.size(predictions)*0.002,0.002)

    plt.plot(time_test, predictions,label="Prediction")
    plt.plot(time_test,test_target_data,'r',label="Target")
    plt.grid()
    plt.legend()
    

    if problem.config["WANDB_LOG"] == True:
        wandb.log({"Test sequence": plt})

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
                if name == "l1.ff.weight" and config["INIT_W2_HALF_NEG"]==True and config["BOUNDS_W1_HALF_NEG"]==True:
                    # Set the last half of the neurons the the negative bound
                    if iteration>=number_of_params/2:
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(0)
                    else:
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(bound_config[name]["high"])
                
                # Check if parameters is the leak_i
                elif name.split(".")[-1] == "leak_i" and config["BOUND_LEAKI_HALF_ZERO"]==True:
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
                         "C":problem.C_matrix,
                         "mean": problem.mean,
                         "sigma":problem.sigma,
                         "stds":problem.stds}
        pickle.dump(test_solutions, pickle_out)
        pickle_out.close()

def create_new_training_set():
    # Insert the test dataset every ... times, otherwise choose a random sequence
    if searcher.step_count%problem.config["SAVE_TEST_SOLUTION_STEPSIZE"] == problem.config["SAVE_TEST_SOLUTION_STEPSIZE"]-1 or searcher.step_count==0 or searcher.steps_count ==problem.config["GENERATIONS"]-1:
        problem.dataset=None
        problem.input_data_new,problem.target_data_new = get_dataset(problem.config,problem.dataset,13)
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
        if problem.C_matrix is None:
            problem.C_matrix = searcher._es.C[np.newaxis, ...]
        else: problem.C_matrix = np.concatenate((problem.C_matrix, searcher._es.C[np.newaxis, ...]), axis=0)

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
        if problem.C_matrix is None:
            problem.C_matrix = searcher.C.detach().numpy()[np.newaxis, ...]
        else: problem.C_matrix = np.concatenate((problem.C_matrix, searcher.C.detach().numpy()[np.newaxis, ...]), axis=0)

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
    problem = LIF_EA_evotorch()
    center_init, std_init = init_conditions(problem)
    
    if problem.config["ALGORITHM"]=="cmaes": searcher = CMAES(problem, stdev_init=1, center_init=center_init, limit_C_decomposition=False, popsize=problem.config["INDIVIDUALS"])
    if problem.config["ALGORITHM"]=="pycma": searcher = PyCMAES(problem,stdev_init=0.4, stdev_max=0.5, center_init=center_init,  popsize=problem.config["INDIVIDUALS"], cma_options={"CMA_stds":std_init}) # Porblem with bound, the mean drifts off, far off the limits ("fixed" with a repair function)

    ### Insert a new dataset every generation
    if problem.config["ANTI_OVERFITTING"]:
        searcher.before_step_hook.append(create_new_training_set)
    if problem.config["SAVE_TEST_SOLUTION_STEPSIZE"] is not None:
        searcher.after_step_hook.append(evaluate_manual_dataset)

    if problem.config["WANDB_LOG"] == True:
        _ = WandbLogger(searcher, project = "SNN_LIF_EA", config=problem.config)
        wandb.config.update({"OS": platform.system()})
        number_first_wandb_name()
    logger = StdOutLogger(searcher)


    searcher.run(problem.config["GENERATIONS"])

    best_solution = searcher.status["best"]
    save_solution(best_solution,problem)
    test_solution(problem,best_solution)
    plot_evolution_parameters(problem)
    plot_stepsize(problem)

