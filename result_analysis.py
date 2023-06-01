import torch
import pygad
import pygad.torchga as torchga
from IZH.SNN_Izh_LI_init import Izhikevich_SNN
from SNN_LIF_LI_init import LIF_SNN
import yaml
from IZH.Izh_LI_EA_PYGAD import get_dataset
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

# for dataset_number in range(10):
sim_time = 40
dataset_number =None # None is the self made 13s dataset
filename = None
folder = "Blimp"
lib = "evotorch"
SNN_TYPE = "LIF"
TIME_STEP = 0.01

create_plots                    = True
plot_with_best_testrun          = True  #True: solution = best performance on manual dataset      False: solution = best performance overall (can be easy dataset)
muliple_test_runs_error_plot    = True  
plot_last_generation            = True

colored_background              = True
spike_count_plot                = True

create_table                    = True
create_csv_file                 = False

plot_sigma                      = False

spectal_analysis                = False
plot_parameters_evolution       = False

#Pick last file name
if filename == None:
    all_files = os.listdir("Results_EA/"+folder)
    splitted_files = [int(f.split("-")[0]) for f in all_files]
    max_ind = splitted_files.index(max(splitted_files))
    filename = all_files[max_ind].split(".")[0] #get rid of .pkl
    print("\nFilename used = ", filename)





#########################################################################################

# if folder.split("/")[0]=="LIF":
#     SNN_TYPE = "LIF"
# else: SNN_TYPE = "IZH"


if lib == "pygad":
    ### load the ga_instance of the last generation
    loaded_ga_instance = pygad.load("Results_EA/"+ folder +"/" + filename)
    loaded_ga_instance.parallel_processing = None
    solution = loaded_ga_instance.best_solutions[-1]

if lib == "evotorch":
    pickle_in = open("Results_EA/"+ folder +"/" + filename+".pkl","rb")
    dict_solutions = pickle.load(pickle_in)
    solution = dict_solutions["best_solution"]

    if plot_with_best_testrun == True:
        solutions = dict_solutions["test_solutions"]
        solutions_error = dict_solutions["error"]
        step_size = dict_solutions["step_size"]
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
        print("Best solutions of the intermidiate testrun implementation is found at ", best_gen, " generations")
    else: print("Solution: Best evaluated solution (note: can be the result of an easy training dataset)")




### Select LIF or IZH mode
if SNN_TYPE == "LIF":
    with open("configs/config_LIF_DEFAULT.yaml","r") as f:
        config = yaml.safe_load(f)

    # Solve for the number of neurons 
    for i in range(100):
        if i**2+6*i+1 == solution.size:
            number_of_neurons = i 
            break
            
    config["NEURONS"] = number_of_neurons
    model = LIF_SNN(None,number_of_neurons)

    #### Initialize neuron states (I, V, spikes) 
    snn_states = torch.zeros(3, 1, model.neurons)
    LI_state = torch.zeros(1,1)

elif SNN_TYPE == "IZH":
    with open("config_Izh_LI_EA.yaml","r") as f:
        config = yaml.safe_load(f)
    model = Izhikevich_SNN(None, "cpu", config)

    ### Initialize neuron states (U, V, spikes) 
    snn_states = torch.zeros(3, 1, model.neurons)
    snn_states[0,:,:] = -20 		#initialize U
    snn_states[1,:,:] = -70			#initialize V
    LI_state = torch.zeros(1,1)

### load in the best parameters in solution
final_parameters =torchga.model_weights_as_dict(model, solution)
thresholds = final_parameters["l1.neuron.thresh"].detach().numpy()

### Get the input and target data
input_data, target_data = get_dataset(config,dataset_number,sim_time)


if config["FITNESS_INCLUDE_DYANMICS"] == False:
    control_input = input_data
    # Make predictions based on the best solution.
    l1_spikes, l1_state, control_output = torchga.predict(model,
                                        solution,
                                        control_input,
                                        snn_states,
                                        LI_state)

    actual_data = control_output[:,0,0]
    target_data = target_data
    mse = torch.nn.MSELoss()
    pearson = PearsonCorrCoef()

    mse_pearson = mse(actual_data, target_data) + (1-pearson(actual_data, target_data))

    actual_data = actual_data.detach().numpy()
    target_data = target_data.detach().numpy()
    print("MSE + Peasrons loss = ", mse_pearson.detach().numpy())

    label_actual_data = "Output of controller"
    title = "Controller output and reference"

if config["FITNESS_INCLUDE_DYANMICS"] == True:
    model_weights_dict = torchga.model_weights_as_dict(model,solution)
    _model = copy.deepcopy(model)
    _model.load_state_dict(model_weights_dict)
    dyn_system = Blimp(config)
    sys_output = torch.tensor([0])
    control_output = np.array([])
    control_input =np.array([])
    input_data = input_data[..., np.newaxis]


    for t in range(input_data.shape[1]):
        ref = input_data[:,t,:,:]
        error = ref - sys_output[-1]
        snn_spikes, snn_states, LI_state = _model(error, snn_states, LI_state)
        snn_states = snn_states[0,:,:,:]    # get rid of the additional list where the items are inserted in forward pass
        LI_state = LI_state[0,:,:]         # get rid of the additional list where the items are inserted in forward pass
        dyn_system.sim_dynamics(LI_state.detach().numpy())
        if t ==0:
            l1_state=snn_states.detach().numpy()[np.newaxis,...]
            l1_spikes=snn_spikes.detach().numpy()
        else:
            l1_state = np.concatenate((l1_state,snn_states.detach().numpy()[np.newaxis,...]))
            l1_spikes = np.concatenate((l1_spikes,snn_spikes.detach().numpy()))
        control_input = np.append(control_input, error.detach().numpy())
        control_output = np.append(control_output, LI_state.detach().numpy())
        sys_output = np.append(sys_output, dyn_system.get_z())
    
    actual_data = sys_output[1:]
    target_data = target_data.detach().numpy()

    mse_loss =(np.square(actual_data -target_data)).mean() # Skip the first 0 height input
    print("MSE = ", mse_loss)

    label_actual_data = "Height of blimp"
    title = "Height control of the Blimp"
    
    time_test = np.arange(0,np.size(actual_data)*config["TIME_STEP"],config["TIME_STEP"])
    plt.plot(time_test, control_output, linestyle = "--", color = "k" )

    # Convert to tensors since it is required for the other part of the script
    l1_state = torch.from_numpy(l1_state)
    l1_spikes = torch.from_numpy(l1_spikes)
    input_data = input_data[0,:,0,0]


time_test = np.arange(0,np.size(actual_data)*config["TIME_STEP"],config["TIME_STEP"])
plt.plot(time_test, actual_data, color = "b", label=label_actual_data)
plt.plot(time_test,target_data, color = 'r',label="Target")
plt.title(title)
plt.grid()
plt.legend()
    






### Calculate spiking sliding window count
spike_count_window = None
spikes_snn = l1_spikes[:,0,:] # spikes_izh of shape (timesteps,neurons)
window_size =100
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

    target_neg = np.clip(target_data,a_min=None, a_max=0)
    target_pos = np.clip(target_data, a_min=0, a_max=None)

    pos_idx_start, pos_idx_end = get_idx_of_non_zero(target_pos, mode="nonzero")
    neg_idx_start, neg_idx_end = get_idx_of_non_zero(target_neg, mode="nonzero")
    zero_idx_start, zero_idx_end = get_idx_of_non_zero(target_data, mode="zero")

    number_of_plots = math.ceil(number_of_neurons/10)
    for idx_plot in range(number_of_plots):
        if idx_plot == number_of_plots-1:
            neurons_in_plot = number_of_neurons - idx_plot*10
        else: neurons_in_plot = 10 
            
        # Start creating the Figure
        time_arr = np.arange(0,sim_time,TIME_STEP)
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
                            axis1[str(row)+","+str(column)].axvspan(pos_idx_start[i]*TIME_STEP, pos_idx_end[i]*TIME_STEP, facecolor="g", alpha= 0.2)
                        for i in range(len(neg_idx_start)):
                            axis1[str(row)+","+str(column)].axvspan(neg_idx_start[i]*TIME_STEP, neg_idx_end[i]*TIME_STEP, facecolor="r", alpha= 0.2)
                        for i in range(len(zero_idx_start)):
                            axis1[str(row)+","+str(column)].axvspan(zero_idx_start[i]*TIME_STEP, zero_idx_end[i]*TIME_STEP, facecolor="k", alpha= 0.2)
                
                ### only plot in U plots
                if column ==1 or column ==3:
                    axis1[str(row)+","+str(column)].axhline(0,linestyle="--",color="k")
                    
                    # Plot the different background, corresponding with target sign
                    if colored_background == True and spike_count_plot==True:
                        for i in range(len(pos_idx_start)):
                            axis1[str(row)+","+str(column)].axvspan(pos_idx_start[i]*TIME_STEP, pos_idx_end[i]*TIME_STEP, facecolor="g", alpha= 0.2)
                        for i in range(len(neg_idx_start)):
                            axis1[str(row)+","+str(column)].axvspan(neg_idx_start[i]*TIME_STEP, neg_idx_end[i]*TIME_STEP, facecolor="r", alpha= 0.2)
                        for i in range(len(zero_idx_start)):
                            axis1[str(row)+","+str(column)].axvspan(zero_idx_start[i]*TIME_STEP, zero_idx_end[i]*TIME_STEP, facecolor="k", alpha= 0.2)
                neuron = neuron +1
            column = 0

        time_arr = np.arange(0,sim_time,TIME_STEP)
        ### Plot the lowest figure
        axis1["input"].plot(time_arr,control_input, label = "Input")
        axis1["input"].plot(time_arr,control_output, label = "Output")
        if config["FITNESS_INCLUDE_DYANMICS"] == False:
            axis1["input"].plot(time_arr,target_data, label = "Target")
            axis1["input"].axhline(0,linestyle="--", color="k")
        axis1["input"].xaxis.grid()
    plt.legend()


    plt.figure()
    ### Plot the lowest figure
    plt.plot(time_arr,control_input, label = "Input")
    plt.plot(time_arr,control_output, label = "Target")
    if config["FITNESS_INCLUDE_DYANMICS"] == False:
        axis1["input"].plot(time_arr,target_data, label = "Target")
        axis1["input"].axhline(0,linestyle="--", color="k")
    plt.grid()
    plt.title("Controller in and output")
    plt.legend()

    # Otherwise the plot and table are shown on the same moment
    if create_table == False and spectal_analysis == False and muliple_test_runs_error_plot==False:
        plt.show()

if create_table == True or create_csv_file==True:
    ######################### plot table ##########################
    round_digits = 3
    neurons = np.array([])
    for neur in range(1,config["NEURONS"]+1):
        neurons = np.append(neurons,str(neur))
    data = neurons

    #Add spike count to the data
    spike_train = l1_state[:,2,0,:].detach().numpy()
    spike_count = np.array([])
    for neuron in range(config["NEURONS"]):
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

    #find row with w1 and w2
    ind_w1 = column_label.index("l1.ff.weight")
    ind_w2 = column_label.index("l2.ff.weight")

    # Swap the w1 to 3 row
    data[[ind_w1,2]] = data[[2,ind_w1]]
    column_label[3], column_label[ind_w2] = column_label[ind_w2], column_label[3]

    # Swap w2 to row 4
    data[[ind_w2,3]] = data[[3,ind_w2]]
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
        'gray': '#AAA9AD',
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
    w1_neg = []
    for w in data[2,:]:
        w = float(w)
        if w<0:
            neg_cel = table[idx,2] #the one correpsonds to w1
            neg_cel.set_facecolor(color_mix["red"])
        idx = idx+1



    idx=1
    w2_neg = []
    for w in data[3,:]:
        w = float(w)
        if w<0:
            neg_cel = table[idx,3] #the 2 correpsonds to w2
            neg_cel.set_facecolor(color_mix["red"])
        idx = idx+1

    if "l1.rec.weight" not in final_parameters:
        plt.show()

    # Create the recurrent table
    if "l1.rec.weight" in final_parameters:
        # data = neurons[..., np.newaxis]
        data = np.round(final_parameters["l1.rec.weight"].detach().numpy(),round_digits)

        norm = plt.Normalize(data.min(), data.max())
        colours = plt.cm.RdYlGn(norm(data))

        column_label = [str(i) for i in range(1,config["NEURONS"]+1)]
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
        
                    
