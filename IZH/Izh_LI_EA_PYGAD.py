import torch
# import pygad.torchga as torchga #Changed al lot of functions in torchga
# import pygad
# from IZH.SNN_Izh_LI_init import Izhikevich_SNN, initialize_parameters
# from wandb_log_functions import create_wandb_summary_table_EA, number_first_wandb_name
# import wandb
# import yaml
import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from datetime import datetime
import platform

# ################################################################
# #                      Define functions
# ################################################################

# def fitness_func(ga_instance, solution, sol_idx):
#     global SNN_izhik, input_data, target_data, loss_function

#     #### Initialize neuron states (u, v, s) 
#     snn_states = torch.zeros(3, 1, SNN_izhik.neurons)
#     snn_states[1,:,:] = -70			#initialize V
#     snn_states[0,:,:] = -20 		#initialize U
#     LI_state = torch.zeros(1,1)


#     izh_output, izh_state, predictions = pygad.torchga.predict(SNN_izhik,
#                                         solution,
#                                         input_data,
#                                         snn_states, #(u, v, s) 
#                                         LI_state) #data in form: input, state_snn, state LI

#     predictions = predictions[:,0,0]

#     abs_error = loss_function(predictions, target_data).detach().numpy() + 0.00000001

#     solution_fitness = 1.0 / abs_error
#     return solution_fitness

# def get_dataset(config, dataset_num, sim_time):
#     if platform.system() == "Linux":
#         prefix = "/scratch/timburgers/SNN_Workspace/"

#     if platform.system() == "Windows":
#         prefix = ""

#     time_step = config["TIME_STEP"]

#     # Either use one of the standard datasets
#     if dataset_num != None:
#         file = "/dataset_"+ str(dataset_num)
#         if config["START_DATASETS_IN_MIDDLE"] == True:
#             start_in_middle = 15*(1/time_step)
#         else: start_in_middle = 1
#     # Or the manual created one
#     else: 
#         file = "/" + config["TEST_DATA_FILE"]
#         start_in_middle=1
    

#     input_data = pd.read_csv(prefix + config["DATASET_DIR"]+ file + ".csv", usecols=config["INPUT_COLUMN_DATAFILE"], header=None, skiprows=start_in_middle, nrows=sim_time*(1/time_step))
#     input_data = torch.tensor(input_data.values).float().unsqueeze(0).unsqueeze(2) 	# convert from pandas df to torch tensor and floats + shape from (seq_len ,features) to (1, seq, feature)


#     target_data = pd.read_csv(prefix + config["DATASET_DIR"] + file + ".csv", usecols=config["LABEL_COLUMN_DATAFILE"], header=None,  skiprows=start_in_middle, nrows=sim_time*(1/time_step))
#     target_data = torch.tensor(target_data.values).float()
#     target_data = target_data[:,0]



#     return input_data, target_data

# def plot_evolution_parameters(best_solutions):
#     generations = np.size(best_solutions,0)
#     final_parameters =torchga.model_weights_as_dict(SNN_izhik, best_solutions[0])

#     ## Initialize dictonary structre{param: [params_gen1, params_gen2 etc..].}
#     full_solution_dict = {key:None for key, _ in final_parameters.items()}



#     ### Fill the dictornary
#     for param in full_solution_dict:
#         for gen in range(generations):
#             gen_parameters =torchga.model_weights_as_dict(SNN_izhik, best_solutions[gen])
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
#         if config["WANDB_LOG"] == True:
#             wandb.log({"Evolution of " + str(param): plt})

#     plt.show()
        
                    
#     return full_solution_dict

# def create_bounds_of_params(torch_ga, limits_config):
#     gene_space = []

#     ### Get the structure and order of the genome
#     param_model =torchga.model_weights_as_dict(torch_ga.model, torch_ga.population_weights[0])

#     ### Check if there is a lim in the config and otherwise add None to it
#     for name, value in param_model.items():
#         number_of_params = len(torch.flatten(value).detach().numpy())
#         for iteration in range(number_of_params):
#             if name in limits_config:
#                 gene_space.append(limits_config[name])
#             else:
#                 gene_space.append(None)
    
#     return gene_space

# def save_ga_instance(ga_instance, wandb_mode, save_flag):
#     if save_flag ==True:
#         # IF wandb is logging --> use wandb file name
#         if wandb_mode == True: 
#             file_name =  wandb.run.name

#         # IF wandb is not logging --> use date and time as file saving
#         else:     
#             date_time = datetime.fromtimestamp(time.time())  
#             file_name = date_time.strftime("%d-%m-%Y_%H-%M-%S")          

#         ga_instance.save(prefix + "results_EA/"+ file_name) 
#     else: return
    


# def main():

#     ###################      Wandb settings     ######################
#     if config["WANDB_LOG"]==True:
#         wandb_mode = "online"	# "online", "offline" or "disabled"
#     else:
#         wandb_mode = "disabled"
    
#     run = wandb.init(project= "SNN_Izhikevich_EA", mode=wandb_mode, reinit=True, config=config)

#     number_first_wandb_name()

#     wandb.config.update({"OS": platform.system()})




#     ####################    Set up EA and run   #########################
#     global target_data
#     # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
#     torch_ga = torchga.TorchGA(model=SNN_izhik,
#                             num_solutions=config["INDIVIDUALS"],
#                             config=config)

#     gene_space = create_bounds_of_params(torch_ga, config["PARAMETER_BOUNDS"])

#     # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
#     ga_instance = pygad.GA(num_generations=config["GENERATIONS"],
#                         num_parents_mating=config["PARENTS_MATING"],
#                         initial_population=torch_ga.population_weights,
#                         fitness_func=fitness_func,
#                         gene_space= gene_space,
#                         save_best_solutions = True,
#                         parallel_processing=["process",config["NUMBER_PROCESSES"]],
#                         parent_selection_type=config["PARENT_SELECTION"],
#                         crossover_type=config["CROSSOVER_TYPE"],
#                         crossover_probability=config["CROSSOVER_PROB"],
#                         mutation_type=config["MUTATION_TYPE"],
#                         mutation_percent_genes=config["MUTATION_PERCENT"])

#     ga_instance.run()


#     ###############   Evaluate outcome    #######################
#     plot_evolution_parameters(ga_instance.best_solutions)
#     save_ga_instance(ga_instance, config["WANDB_LOG"], config["SAVE_LAST_SOLUTION"])

#     # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
#     fit_plot = ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

#     if config["WANDB_LOG"] == True:
#         wandb.log({"Fitness over generations":fit_plot})

#     # Returning the details of the best solution.
#     solution, solution_fitness, solution_idx = ga_instance.best_solution()
#     print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
#     print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

#     ### Print the final parameters to the terminal
#     final_parameters =torchga.model_weights_as_dict(SNN_izhik, solution)
#     for key, value in final_parameters.items():
#         print(key, value)


#     #################    Test sequence       ############################
#     #### Initialize neuron states (u, v, s) 
#     snn_states = torch.zeros(3, 1, SNN_izhik.neurons)
#     snn_states[1,:,:] = -70			#initialize V
#     snn_states[0,:,:] = -20 		#initialize U
#     LI_state = torch.zeros(1,1)

#     # Make predictions based on the best solution.
#     izh_output, izh_state, predictions = pygad.torchga.predict(SNN_izhik,
#                                         solution,
#                                         input_data,
#                                         snn_states,
#                                         LI_state)
#     predictions = predictions[:,0,0]
    
#     spike_train = izh_state[:,2,0,:].detach().numpy()
#     create_wandb_summary_table_EA(run, spike_train, config, final_parameters)


#     print("\n Predictions : \n", predictions.detach().numpy())
#     global target_data

#     abs_error = loss_function(predictions, target_data)
#     print("Absolute Error : ", abs_error.detach().numpy())


#     predictions = predictions.detach().numpy()
#     target_data = target_data.detach().numpy()
#     time_test = np.arange(0,np.size(predictions)*0.002,0.002)

#     plt.plot(time_test, predictions,label="Prediction")
#     plt.plot(time_test,target_data,'r',label="Target")
#     plt.grid()
#     plt.legend()
    

#     if config["WANDB_LOG"] == True:
#         wandb.log({"Test sequence": plt})

#     plt.show()


# #############################   Set certain global variables          #############################
# This is necessary when multiprocessing, since the fitness function need certain global parameters 
# that can not be accessed as input variables, thus this part of the code is constantly runned when
# a new process is spawned

# if platform.system() == "Linux":
#     prefix = "/scratch/timburgers/SNN_Workspace/"

# if platform.system() == "Windows":
#     prefix = ""


# ### Read config file
# with open(prefix + "IZH/config_Izh_LI_EA.yaml","r") as f:
#     config = yaml.safe_load(f)
# device = "cpu"

# ### Initialize SNN + Dataset
# param_init = initialize_parameters(config)
# SNN_izhik = Izhikevich_SNN(param_init, device, config).to(device) 
# input_data, target_data = get_dataset(config, config["DATASET_NUMBER"], config["SIM_TIME"])
# loss_function = torch.nn.MSELoss()

# if __name__ == '__main__':
#     main()


