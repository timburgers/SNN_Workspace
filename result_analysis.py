import torch
import pygad
import pygad.torchga as torchga
from IZH.SNN_Izh_LI_init import Izhikevich_SNN
from SNN_LIF_LI_init import LIF_SNN
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
sim_time = 1
dataset_number = None                                                    # None is the test_dataset
filename = 183                                                          #None --> highest number, or int or str (withou .pkl)
folder_of_model = "Blimp"                                               # all folder under the folder Results_EA
lib_algorithm = "evotorch"                                              # evotorch or pygad
SNN_TYPE = "LIF"                                                        # either LIF or IZH
window_size =1
# config["DATASET_DIR"] = "Sim_data/height_control_PID/fast_steps"

####################
exclude_non_spiking_neurons = True
# excluded_neurons = [3,11,19,20,21,22] #idx start at 1
excluded_neurons=[]
new_dataset = "sine_derivative_large"
new_dataset_number = 0



create_plots                    = True
plot_with_best_testrun          = True  #True: solution = best performance on manual dataset      False: solution = best performance overall (can be easy dataset)
muliple_test_runs_error_plot    = False  
plot_last_generation            = False

colored_background              = True
spike_count_plot                = True
create_table                    = True


create_csv_file                 = False
plot_sigma                      = False
spectal_analysis                = False
plot_parameters_evolution       = False


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

####### Functions
# Add the shared weight in the coorect way to the solutions
def full_parameter_list(final_parameters):
    for name, param in final_parameters.items():
        if torch.flatten(param).shape[0] == number_of_neurons/2:
            if name =="l1.ff.weight":
                final_parameters[name] = torch.flatten(torch.cat((param,-1*param),dim=1))
            if name == "l2.ff.weight":
                final_parameters[name] = torch.flatten(torch.transpose(torch.cat((param,-1*param)),0,1))
            if name =="l1.ff.bias" or name=="l1.neuron.leak_i":
                final_parameters[name] = torch.flatten(torch.stack((param,param),dim=1))
    return final_parameters       

# Run the simulation
def run_sim(fitness_mode, config, controller,solution,input_dataset, save=True):
    #The solution is defined in a numpy array shape

    snn_states = torch.zeros(3, 1, controller.neurons) # frist dim in order [current,mempot,spike]
    LI_state = torch.zeros(1,1)
    sys_output = None

    final_parameters =torchga.model_weights_as_dict(controller, solution)
    controller.load_state_dict(final_parameters)

    if  fitness_mode== 1: #Only controller simlation
        fitness_measured, control_state = run_controller(controller,input_dataset, snn_states,LI_state, save_mode=save)
        control_output = fitness_measured
        control_input = torch.flatten(input_dataset).detach().numpy()

    elif fitness_mode == 2 or fitness_mode == 3:#Also simulate the dyanmics
        fitness_measured, control_input, control_state, control_output = run_controller_dynamics(config,controller,input_dataset, snn_states,LI_state, save_mode=save)
        sys_output = fitness_measured

    return fitness_measured, control_input, control_state ,control_output ,final_parameters

def torch_delete(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

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
     
    # with open("configs/config_LIF_DEFAULT.yaml","r") as f:
    #     config = yaml.safe_load(f)
    config = dict_solutions["config"]
    number_of_neurons = config["NEURONS"]
    controller = LIF_SNN(None,number_of_neurons, config["LAYER_SETTING"])

    # #### Initialize neuron states (I, V, spikes) 
    # snn_states = torch.zeros(3, 1, controller.neurons)
    # LI_state = torch.zeros(1,1)

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
input_data, fitness_target = get_dataset(config, dataset_number, sim_time)
fitness_mode = config["TARGET_FITNESS"]

#Override TARGET_FITNESS in config dict to load ideal pid response
config["TARGET_FITNESS"] = 1 #since the target of mode 1 is the pid response
_, ideal_pid_response = get_dataset(config, dataset_number, sim_time)

##################           RUN SIM                ########################################################
fitness_measured, control_input, control_state ,control_output ,final_parameters = run_sim(fitness_mode, config,controller,solution,input_data,True)

######################################################
if excluded_neurons or exclude_non_spiking_neurons:
    if excluded_neurons: #if there are entrys
        excluded_neurons = [i - 1 for i in excluded_neurons] #since ind start at zero for further calulcations

    # Check which neurons are non-spiking
    if exclude_non_spiking_neurons ==True:
        for neuron in range(controller.neurons):
            if not control_state[:,2,0,neuron].any():
                excluded_neurons.append(neuron)

    # Pre process the second run with the selected neurons only
    all_params = full_parameter_list(final_parameters)
    sparse_params = copy.deepcopy(all_params)
    new_number_of_neurons = number_of_neurons-len(excluded_neurons)
    sparse_np_solution = np.array([])
    for name, param in all_params.items():
        sparse_params = param
        if torch.flatten(param).shape[0] == number_of_neurons:
            sparse_params=torch_delete(param,excluded_neurons)
        sparse_np_solution = np.append(sparse_np_solution,sparse_params)


    sparse_config = copy.deepcopy(config)
    sparse_config["NEURONS"] = new_number_of_neurons
    sparse_config["LAYER_SETTING"]["l1"]["shared_leak_i"]          = False
    sparse_config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"] = False
    sparse_config["LAYER_SETTING"]["l2"]["shared_weight_and_bias"] = False
    sparse_config["DATASET_DIR"] = "Sim_data/height_control_PID/" + new_dataset

    sparse_controller = LIF_SNN(None,new_number_of_neurons, sparse_config["LAYER_SETTING"])
    new_input_data, fitness_target = get_dataset(sparse_config, new_dataset_number, sim_time)

    fitness_measured, control_input, control_state ,control_output ,final_parameters = run_sim(fitness_mode, sparse_config,sparse_controller,sparse_np_solution,new_input_data, True)
    controller = sparse_controller
    number_of_neurons = new_number_of_neurons
    ideal_pid_response = fitness_target
#####################################################

# Calculate the fitness value
fitness_value = evaluate_fitness(fitness_mode, fitness_measured, fitness_target)
print("Fitness value = ", np.round(fitness_value.item(),5))

# calculate the splitted 
mse = torch.nn.MSELoss()
pearson = PearsonCorrCoef()
#Evaluate fitness using MSE and additionally pearson if there should be a linear correlation between target and output
_fitness_target = torch.flatten(fitness_target)
_fitness_measured = torch.from_numpy(fitness_measured)
mse = mse(_fitness_measured,_fitness_target)
print("MSE = ", np.round(mse.item(),5))
if fitness_mode == 1 or fitness_mode == 3:
    pearson =  (1-pearson(_fitness_measured,_fitness_target)) #pearson of 1 means linear correlation
    print("Pearson = ", np.round(pearson.item(),5))



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
plt.title(title)
plt.grid()
plt.legend()
if create_plots == False:
    plt.show()


# Convert to tensors since it is required for the other part of the script
l1_state = torch.from_numpy(control_state)
l1_spikes = torch.from_numpy(control_state[:,:,0,:])





### Calculate spiking sliding window count
spike_count_window = None
spikes_snn = l1_spikes[:,2,:] # spikes_izh of shape (timesteps,neurons)

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

    input_neg = np.clip(control_input,a_min=None, a_max=0)
    input_pos = np.clip(control_input, a_min=0, a_max=None)

    pos_idx_start, pos_idx_end = get_idx_of_non_zero(input_pos, mode="nonzero")
    neg_idx_start, neg_idx_end = get_idx_of_non_zero(input_neg, mode="nonzero")
    zero_idx_start, zero_idx_end = get_idx_of_non_zero(input_data, mode="zero")

    thresholds = final_parameters["l1.neuron.thresh"].detach().numpy()
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


        recov_or_current = l1_state[:,0,0,:].detach().numpy()
        mem_pot = l1_state[:,1,0,:].detach().numpy() 

        ### plot the raster of U and V of the 10 neurons (5xV, 5xU, 5xV, 5xU)
        for column in range(2):
            neuron = 0+10*idx_plot
            for row in range(neurons_in_plot):
                
                if column ==0 or column ==2:
                    y = mem_pot[:,neuron]

                # Plot in the second column
                if column ==1 or column ==3:
                    if spike_count_plot== True:
                        y=spike_count_window[neuron,:]

                    else: 
                        y = recov_or_current[:,neuron] #LIF --> current and for IZH --> recovery variable
                
                if row==5:
                    column = column + 2
                if row>4:
                    row = row -5
                
                axis1[str(row)+","+str(column)].plot(time_arr,y)

                axis1[str(row)+","+str(column)].xaxis.grid()

                ### only plot in V plots
                if column ==0 or column ==2:
                    axis1[str(row)+","+str(column)].axhline(thresholds[neuron],color="r")

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
        axis1["input"].plot(time_arr,control_input, label = "SNN input")
        axis1["input"].plot(time_arr,control_output, label = "SNN output")
        axis1["input"].plot(time_arr,ideal_pid_response, label = "PID reponse")
        axis1["input"].axhline(0,linestyle="--", color="k")
        axis1["input"].xaxis.grid()
    plt.legend()


    plt.figure()
    ### Plot the lowest figure
    plt.plot(time_arr,control_input, label = "SNN input")
    plt.plot(time_arr,control_output, label = "SNN output")
    plt.plot(time_arr,ideal_pid_response, label = "PID reponse")
    plt.axhline(0,linestyle="--", color="k")
    plt.grid()
    plt.title("Controller response")
    plt.legend()

    # Otherwise the plot and table are shown on the same moment
    if create_table == False and spectal_analysis == False and muliple_test_runs_error_plot==False:
        plt.show()

if create_table == True or create_csv_file==True:
    ######################### plot table ##########################
    round_digits = 3
    neurons = np.array([])
    for neur in range(0,config["NEURONS"]):
        if neur not in excluded_neurons:
            neurons = np.append(neurons,str(neur+1))
    data = neurons

    #Add spike count to the data
    spike_train = l1_state[:,2,0,:].detach().numpy()
    spike_count = np.array([])
    for neuron in range(number_of_neurons):
        spikes = float(np.sum(spike_train[:,neuron]))
        spike_count = np.append(spike_count,spikes)
    data = np.vstack((data,spike_count))

    column_label = ["Neuron", "Spike count"]

    # Add all trained parameters to the data array
    for parameter in final_parameters.keys():
        data_param = np.round(torch.flatten(final_parameters[parameter]).detach().numpy(),round_digits)
        # Only add the parameters which has one param per neurons( so not leak l2 and rec connections)
        if data_param.size == neurons.size:
            data =np.vstack((data,data_param))
            column_label.append(parameter)
        else: print("Parameter named ", parameter, " is not included in the table")

    #find row with w1 and w2 and swap them such they are in the beginning of the table
    ind_w1 = column_label.index("l1.ff.weight")
    ind_w2 = column_label.index("l2.ff.weight")
    data[[ind_w1,2]] = data[[2,ind_w1]]
    column_label[3], column_label[ind_w2] = column_label[ind_w2], column_label[3]     # Swap the w1 to 3 row
    data[[ind_w2,3]] = data[[3,ind_w2]]                                               # Swap w2 to row 4
    column_label[2], column_label[ind_w1] = column_label[ind_w1], column_label[2]
    
if create_table ==True:
    plt.figure(linewidth=1,
            tight_layout={"pad":1})
    table = plt.table(cellText=np.transpose(data), colLabels=column_label, loc='center')

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(15)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Hide axes border
    plt.box(on=None)

    # Colors for highlighting
    color_mix = {
        'white':'#FFFFFF',
        'gray': '#D3D3D3',
        'black':'#313639',
        'purple':'#AD688E',
        'orange':'#D18F77',
        'yellow':'#E8E190',
        'ltgreen':'#CCD9C7',
        'dkgreen':'#96ABA0',
        'red':'#FFCCCB',
        }



    # highlight w1 cells
    idx=1
    for w in data[2,:]:
        w = float(w)
        if w<0:
            table[idx,2].set_facecolor(color_mix["red"])
        idx = idx+1

    idx=1
    for w in data[3,:]:
        w = float(w)
        if w<0:
            table[idx,3].set_facecolor(color_mix["red"])
        idx = idx+1

    #greyout the non spiking neurons (NOTE: data is later inversed, sot it is column,row now)
    row=1
    for spike_count in data[1,:]:
        if float(spike_count) == 0:
            for col in range(len(data)):
                table[row,col].set_facecolor(color_mix["gray"])
        row +=1




    if "l1.rec.weight" not in final_parameters:
        plt.show()

    # Create the recurrent table
    if "l1.rec.weight" in final_parameters:
        # data = neurons[..., np.newaxis]
        data = np.round(final_parameters["l1.rec.weight"].detach().numpy(),round_digits)

        norm = plt.Normalize(data.min(), data.max())
        colours = plt.cm.RdYlGn(norm(data))

        column_label = [str(i) for i in range(1,number_of_neurons+1)]
        plt.figure(linewidth=1,
                tight_layout={"pad":1})
        table = plt.table(cellText=data, colLabels=column_label, rowLabels=column_label, cellColours=colours, loc='center')

        # Set font size
        table.auto_set_font_size(False)
        table.set_fontsize(15)

        # Hide axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Hide axes border
        plt.box(on=None)
        plt.show()

if create_csv_file == True:
    ### Create a matlab file with parameters
    data_mat = np.transpose(np.vstack((w1,w2,thres,a,b,c,d,v2,v1,v0,utau)))
    np.savetxt("test_matlab.csv", data_mat)


if spectal_analysis == True:
    N = len(predictions)
    T= 1./500.
    x = np.linspace(0,N*T,N,endpoint=False)
    y = predictions

    yf= fft(y)
    xf= fftfreq(N,T)[:N//2]

    plt.figure()
    plt.plot(xf,2.0/N*np.abs(yf[0:N//2]))
    plt.grid()
    plt.show()

if muliple_test_runs_error_plot == True:
    plt.figure()
    gen_arr = np.arange(0,len(solutions_error)*step_size,step_size)
    # if generations%step_size!=0:
    #     gen_arr = np.append(gen_arr,generations)
    plt.plot(gen_arr,solutions_error)

    plt.title("Error of Manual Test Run every "+ str(step_size) +" generations")
    plt.xlabel("Generations [-]")
    plt.ylabel("Error [-]")
    plt.grid()
    if plot_sigma == False:
        plt.show()

if plot_sigma == True:
    sigma = dict_solutions["sigma"]
    gen_arr = np.arange(0, len(sigma))
    plt.figure()
    plt.title("Stepsize over generations")
    plt.xlabel("Generations [-]")
    plt.ylabel("Stepsize [-]")
    plt.plot(gen_arr,sigma)
    plt.grid()
    plt.show()

if plot_parameters_evolution == True:

    generations = np.size(dict_solutions["mean"],0)
    parameters_mean = dict_solutions["mean"] #shape (generations, parameters)
    final_parameters =torchga.model_weights_as_dict(model, parameters_mean[0,:])

    ## Initialize dictonary structre{param: [params_gen1, params_gen2 etc..].}
    full_solution_dict = {key:None for key, _ in final_parameters.items()}



    ### Fill the dictornary
    for param in full_solution_dict:
        for gen in range(generations):
            gen_parameters =torchga.model_weights_as_dict(model, parameters_mean[gen,:])
            for name, value in gen_parameters.items():
                if param == name:
                    value = torch.flatten(value).detach().numpy()
                    if full_solution_dict[param] is None: #Check is dict is empty 
                        full_solution_dict[param] = value
                    else:
                        full_solution_dict[param] = np.vstack((full_solution_dict[param],value))

                    break


    ### Plot the different diagrams
    gen_arr = np.arange(0,generations)
    for param in full_solution_dict:
        plt.figure()
        for num_param in range(len(full_solution_dict[param][0,:])):
            param_per_gen = full_solution_dict[param][:,num_param]
            param_per_gen = param_per_gen.flatten()
            plt.plot(gen_arr,param_per_gen,label=str(num_param))
        plt.title("Evolution of " + str(param))
        plt.xlabel("Generations [-]")
        plt.xticks(gen_arr)
        plt.legend()
        plt.grid()
    plt.show()
        
                    
