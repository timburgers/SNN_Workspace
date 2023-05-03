import torch
import pygad
import pygad.torchga as torchga
from SNN_Izh_LI_init import Izhikevich_SNN, initialize_parameters
import yaml
from Izh_LI_EA_PYGAD import get_dataset
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.fft import fft, fftfreq


sim_time = 30
dataset_number = 3 # None is the self made 13s dataset
filename = "pretty-darkness-189"
lib = "evotorch"

create_plots = True
colored_background = False
create_table = False
create_csv_file = False

spectal_analysis = True











#########################################################################################

if lib == "pygad":
    ### load the ga_instance of the last generation
    loaded_ga_instance = pygad.load("Results_EA/"+ filename)
    loaded_ga_instance.parallel_processing = None
    solution = loaded_ga_instance.best_solutions[-1]

if lib == "evotorch":
    pickle_in = open("Results_EA/Evotorch/" + filename+".pkl","rb")
    solution = pickle.load(pickle_in)
    solution = solution.values.detach().numpy()

### Read config file and insert dummy data input the SNN
with open("config_Izh_LI_EA.yaml","r") as f:
    config = yaml.safe_load(f)
device = "cpu"
param_init = initialize_parameters(config)
SNN_izhik = Izhikevich_SNN(param_init, device, config)


#### Initialize neuron states (u, v, s) 
snn_states = torch.zeros(3, 1, SNN_izhik.neurons)
snn_states[1,:,:] = -70			#initialize V
snn_states[0,:,:] = -20 		#initialize U
LI_state = torch.zeros(1,1)

### load in the best parameters in solution

final_parameters =torchga.model_weights_as_dict(SNN_izhik, solution)
thresholds = final_parameters["l1.neuron.thresh"].detach().numpy()

### Get the input and target data
input_data, target_data = get_dataset(config,dataset_number,sim_time)

# Make predictions based on the best solution.
izh_output, izh_state, predictions = pygad.torchga.predict(SNN_izhik,
                                    solution,
                                    input_data,
                                    snn_states,
                                    LI_state)

predictions = predictions[:,0,0].detach().numpy()
target_data = target_data.detach().numpy()
input_data = input_data.detach().numpy()
input_data = input_data[0,:,0]
izh_u = izh_state[:,0,0,:].detach().numpy()
izh_v = izh_state[:,1,0,:].detach().numpy()

    

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

    # Start creating the Figure
    time_arr = np.arange(0,sim_time,0.002)
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

    # # plt.figure()
    # fig,axis1 = plt.subplots(6,4, sharex=True)

    ### plot the raster of U and V of the 10 neurons (5xV, 5xU, 5xV, 5xU)
    for column in range(2):
        neuron = 0
        for row in range(10):
            
            if column ==0 or column ==2:
                y = izh_v[:,neuron]
            if column ==1 or column ==3:
                y = izh_u[:,neuron]
            
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
                        axis1[str(row)+","+str(column)].axvspan(pos_idx_start[i]*0.002, pos_idx_end[i]*0.002, facecolor="g", alpha= 0.2)
                    for i in range(len(neg_idx_start)):
                        axis1[str(row)+","+str(column)].axvspan(neg_idx_start[i]*0.002, neg_idx_end[i]*0.002, facecolor="r", alpha= 0.2)
                    for i in range(len(zero_idx_start)):
                        axis1[str(row)+","+str(column)].axvspan(zero_idx_start[i]*0.002, zero_idx_end[i]*0.002, facecolor="k", alpha= 0.2)
            
            ### only plot in U plots
            if column ==1 or column ==3:
                axis1[str(row)+","+str(column)].axhline(0,linestyle="--",color="k")
            neuron = neuron +1
        column = 0

    ### Plot the lowest figure
    axis1["input"].plot(time_arr,input_data, label = "Input")
    axis1["input"].plot(time_arr,target_data, label = "Target")
    axis1["input"].plot(time_arr,predictions, label = "Output")
    axis1["input"].axhline(0,linestyle="--", color="k")
    axis1["input"].xaxis.grid()


    plt.legend()


    plt.figure()
    ### Plot the lowest figure
    plt.plot(time_arr,input_data, label = "Input")
    plt.plot(time_arr,target_data, label = "Target")
    plt.plot(time_arr,predictions, label = "Output")
    plt.axhline(0,linestyle="--", color="k")
    plt.grid()
    plt.legend()

    # Otherwise the plot and table are shown on the same moment
    if create_table == False and spectal_analysis == False:
        plt.show()

######################### plot table ##########################
round_digits = 3
neurons = np.array(["1","2","3","4","5","6","7","8","9","10"])
w1 =np.round(torch.flatten(final_parameters["l1.ff.weight"]).detach().numpy(),round_digits)
w2 =np.round(torch.flatten(final_parameters["l2.ff.weight"]).detach().numpy(),round_digits)
thres = np.round(final_parameters["l1.neuron.thresh"].detach().numpy(),round_digits)
a =np.round(final_parameters["l1.neuron.a"].detach().numpy(),round_digits)
b =np.round(final_parameters["l1.neuron.b"].detach().numpy(),round_digits)
c =np.round(final_parameters["l1.neuron.c"].detach().numpy(),round_digits)
d =np.round(final_parameters["l1.neuron.d"].detach().numpy(),round_digits)
v2 =np.round(final_parameters["l1.neuron.v2"].detach().numpy(),round_digits)
v1 =np.round(final_parameters["l1.neuron.v1"].detach().numpy(),round_digits)
v0 =np.round(final_parameters["l1.neuron.v0"].detach().numpy(),round_digits)
utau =np.round(final_parameters["l1.neuron.tau_u"].detach().numpy(),round_digits)

if create_table ==True:
    ### calculate spike count
    spike_train = izh_state[:,2,0,:].detach().numpy()
    spike_count = np.array([])
    for neuron in range(config["NEURONS"]):
        spikes = float(np.sum(spike_train[:,neuron]))
        spike_count = np.append(spike_count,spikes)


    # Create data
    data = np.transpose(np.vstack((neurons,spike_count,w1,w2,thres,a,b,c,d,v2,v1,v0,utau)))
    columns = ("Neuron","Spike count","W1", "W2", "Thres","a","b","c","d","v2","v1","v0","u tau")


    plt.figure(linewidth=1,
            tight_layout={"pad":1})
    table = plt.table(cellText=data, colLabels=columns, loc='center')

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
    for w in w1:
        if w<0:
            neg_cel = table[idx,2] #the one correpsonds to w1
            neg_cel.set_facecolor(color_mix["red"])
        idx = idx+1



    idx=1
    w2_neg = []
    for w in w2:
        if w<0:
            neg_cel = table[idx,3] #the 2 correpsonds to w2
            neg_cel.set_facecolor(color_mix["red"])
        idx = idx+1

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

