
import torch
import pygad.torchga as torchga
from SNN_Izh_LI_init import Izhikevich_SNN, initialize_parameters
from Izh_LI_EA_PYGAD import get_dataset
import yaml
import numpy as np
import matplotlib.pyplot as plt
import pickle
# import os
# os.environ["RAY_DEDUP_LOGS"]= "0"
from evotorch import SolutionBatch
from random import randint

from evotorch import Problem
from evotorch.algorithms import CMAES, PyCMAES
from evotorch.logging import StdOutLogger,WandbLogger
import ray
import wandb
from datetime import datetime
import time

import warnings
warnings.filterwarnings("ignore")
ray.init(log_to_driver=False)

class izh_EA_evotorch(Problem):
    def __init__(self):
        ### Read config file
        with open("config_EVOTORCH.yaml","r") as f:
            self.config = yaml.safe_load(f)

        self.SNN_izhik = Izhikevich_SNN(None, "cpu", self.config)
        self.input_data, self.target_data = get_dataset(self.config, self.config["DATASET_NUMBER"], self.config["SIM_TIME"])
        self.loss_function = torch.nn.MSELoss()
        self.bounds = create_bounds(self.SNN_izhik, self.config)


        super().__init__(
            objective_sense="min",
            solution_length=111,
            initial_bounds=(-1,1),
            num_actors=self.config["NUMBER_PROCESSES"],
            bounds=self.bounds
        )


    def _evaluate(self, solution):
        #### Initialize neuron states (u, v, s) 
        snn_states = torch.zeros(3, 1, self.SNN_izhik.neurons)
        snn_states[1,:,:] = -70			#initialize V
        snn_states[0,:,:] = -20 		#initialize U
        LI_state = torch.zeros(1,1)
        solution_np = solution.values.detach().numpy()

        izh_output, izh_state, predictions = torchga.predict(self.SNN_izhik,
                                            solution_np,
                                            self.input_data,
                                            snn_states, #(u, v, s) 
                                            LI_state) #data in form: input, state_snn, state LI

        predictions = predictions[:,0,0]
        solution_fitness = self.loss_function(predictions, self.target_data).detach().numpy()
        print(solution_fitness)
        solution.set_evals(solution_fitness)
    

# Get the intiail position of the center and of the step size
def init_conditions(problem):
    init_config = problem.config["INITIAL_PARAMS_RANDOM"]
    bounds_config= problem.config["PARAMETER_BOUNDS"] #The bounds are usedto scale the initial step size
    model = problem.SNN_izhik
    names_center_init = np.array([])
    center_init = np.array([])
    std_init=np.array([])

    ### Get the structure and order of the genome
    param_model =torchga.model_weights_as_dict(model, np.ones(111))

    ### Check if there is a lim in the config and otherwise add None to it
    for name, value in param_model.items():
        number_of_params = len(torch.flatten(value).detach().numpy())
        for iteration in range(number_of_params):
            
            # Fill in initial condition of mu
            if name in init_config:
                center_init = np.append(center_init,init_config[name])
                if problem.config["HALF_NEGATIVE_WEIGHTS"]==True and name.split(".")[-1] == "weight":
                    if iteration>=number_of_params/2:
                        center_init[-1]=-center_init[-1]

            else:
                center_init = np.append(center_init,init_config[name])
            names_center_init = np.append(names_center_init,name)

            # Fill in initial condition of stepsize (std)
            initial_step_size = (bounds_config[name]["high"]-bounds_config[name]["low"])/10
            std_init = np.append(std_init,initial_step_size)

    return center_init, std_init

def test_solution(problem, solution):
    # Initialize varibales from problem Class
    model = problem.SNN_izhik
    test_input_data, test_target_data = get_dataset(problem.config, problem.config["DATASET_NUMBER"], problem.config["SIM_TIME"])
    solution = solution.values.detach().numpy()

    ### Print the final parameters to the terminal
    final_parameters =torchga.model_weights_as_dict(model, solution)
    for key, value in final_parameters.items():
        print(key, value)


    #################    Test sequence       ############################
    #### Initialize neuron states (u, v, s) 
    snn_states = torch.zeros(3, 1, model.neurons)
    snn_states[1,:,:] = -70			#initialize V
    snn_states[0,:,:] = -20 		#initialize U
    LI_state = torch.zeros(1,1)

    # Make predictions based on the best solution.
    izh_output, izh_state, predictions = torchga.predict(model,
                                        solution,
                                        test_input_data,
                                        snn_states,
                                        LI_state)
    predictions = predictions[:,0,0]
    
    spike_train = izh_state[:,2,0,:].detach().numpy()
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

def create_bounds(model,config):
    bound_config = config["PARAMETER_BOUNDS"]

    lower_bounds = []
    upper_bounds = []
    ### Get the structure and order of the genome
    param_model =torchga.model_weights_as_dict(model, np.ones(111)) # fill with dummy inputs

    ### Check if there is a lim in the config and otherwise add None to it
    for name, value in param_model.items():
        number_of_params = len(torch.flatten(value).detach().numpy())
        for iteration in range(number_of_params):
            if name in bound_config:

                # Check if parameters is the weights
                if name.split(".")[-1] == "weight":

                    # Set the first 5 neurons to a positive bound 
                    if iteration<number_of_params/2:
                        lower_bounds.append(0)
                        upper_bounds.append(bound_config[name]["high"])

                    # Set the last 5 neurons the the negative bound
                    else:
                        lower_bounds.append(bound_config[name]["low"])
                        upper_bounds.append(0)
                else:
                    lower_bounds.append(bound_config[name]["low"])
                    upper_bounds.append(bound_config[name]["high"])
            else:
                lower_bounds.append(None)
                upper_bounds.append(None)

    return (lower_bounds , upper_bounds)

def save_solution(solution, problem):
    wandb_mode = problem.config["WANDB_LOG"]
    save_flag = problem.config["SAVE_LAST_SOLUTION"]
    
    if save_flag == True:
        if wandb_mode == True: 
            file_name =  wandb.run.name

        # IF wandb is not logging --> use date and time as file saving
        else:     
            date_time = datetime.fromtimestamp(time.time())  
            file_name = date_time.strftime("%d-%m-%Y_%H-%M-%S")      

        pickle_out = open("Results_EA/Evotorch/"+ file_name+ ".pkl","wb")
        pickle.dump(solution, pickle_out)
        pickle_out.close()

def create_new_training_set(batch: SolutionBatch):
    dataset = randint(0,99)
    problem.input_data, problem.target_data = get_dataset(problem.config, dataset, problem.config["SIM_TIME"])


if __name__ == "__main__":
    problem = izh_EA_evotorch()
    if problem.config["ANTI_OVERFITTING"]:
        problem.before_eval_hook.append(create_new_training_set)
    center_init, std_init = init_conditions(problem)
    
    # searcher = CMAES(problem, stdev_init=0.05, center_init=center_init, limit_C_decomposition=False, popsize=20)
    searcher = PyCMAES(problem,stdev_init=1, center_init=center_init,  popsize=problem.config["INDIVIDUALS"], cma_options={"CMA_stds":std_init})
    searcher.before_step_hook
    if problem.config["WANDB_LOG"] == True:
        _ = WandbLogger(searcher, project = "SNN_Izhikevich_EA", config=problem.config)
    logger = StdOutLogger(searcher)

    

    searcher.run(problem.config["GENERATIONS"])

    best_solution = searcher.status["best"]
    save_solution(best_solution,problem)
    test_solution(problem,best_solution)

