import torch
#from Encoding import input_cur
import numpy as np
import matplotlib.pyplot as plt
import math
from spiking.torch.neurons.base import BaseNeuron
from spiking.torch.layers.linear import BaseLinear

# output = input_cur("spike", 25000,1,0.1)

# t = np.arange(0,2500,0.1)

def sliding_window(spike_train, time_step, window_size, window_step):
    """
    spike train = the output spike train that will be decoded
    time_step   = the time step used in the data
    window_size = lenght in ms of the sliding window
    windor_step = stepsize in ms used for sliding the window
    """
    # Shape (num_data_points, neurons)
    decoded_output = np.zeros([math.trunc(((len(spike_train[:,0])*time_step)-window_size)/window_step),np.size(spike_train,axis=1)])
    # print("decoded output shape = ", np.shape(decoded_output))
    # Loop over number of outpur spike trains
    for neuron in range(np.size(spike_train,axis=1)):

        # convert to numpy array from torch tensor
        spike_train_i = spike_train[:,neuron]

        # Get sequence length (total datapoint in spiketrain) and time in ms
        seq_len = len(spike_train_i)
        seq_time = seq_len*time_step

        # Calcuate how many time the window can be fitted in the data
        num_data_points = math.trunc((seq_time - window_size)/window_step)

        #initalize list
        spike_count_window = np.array([])

        #loop over all windows an calculate the spike count
        for i in range(num_data_points):
            idx_start = int((i*window_step)/time_step)
            idx_end = int((i*window_step + window_size)/time_step)
            spike_train_window = spike_train_i[idx_start:idx_end]
            spike_count = np.sum(spike_train_window)
            spike_count_window = np.append(spike_count_window,spike_count)

       
        decoded_output[:,neuron] = spike_count_window

    time_window = np.arange(window_size/2,window_size/2+num_data_points*window_step,window_step)

    return time_window, decoded_output



####################################################################################################################################################################



# time, count = sliding_window(output,0.1,100,20)
# count = count/np.max(count)
# print(time.size)
# print(count.size)
# plt.plot(t,output)
# plt.plot(time,count,color = 'r')
# plt.show()