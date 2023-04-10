import random
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def input_cur(type, sim_time, time_step, current_value,input_neurons,step_percentage): 
	"""
	Type options: spike, constant, triangle, step or webb
	"""
	data_points = int(sim_time/time_step)
	if (type == "spike"):
		# Generate random spike pattern
		percentage_of_spikes = 0.002
		spike_list = [0]*int(data_points*(1-percentage_of_spikes))+[current_value]*int(data_points*percentage_of_spikes)
		random.shuffle(spike_list)
		spike_list = torch.FloatTensor(spike_list)
		spike_input = torch.reshape(spike_list,(data_points,1))
		return spike_input

	if (type == "constant"):
		# Set constant input current
		const_input = torch.zeros(data_points,1)
		for i in range(data_points):
			const_input[i,0]=current_value
		return const_input

	if (type == "triangle"):
		# Set constant input current
		triangle_input = torch.zeros(data_points,1)
		for i in range(data_points):
			if i < (data_points/2): triangle_input[i,0]=current_value*i
			else: triangle_input[i,0]=current_value*(data_points - i)
		return triangle_input

	if (type == "step"):

		step_input = torch.zeros(data_points,1)
		for i in range(data_points):
			if i < (data_points*step_percentage): step_input[i,0]=0
			else: step_input[i,0]=current_value#current_value
		return step_input
	
	if (type == "webb_test"):
		webb_input = torch.zeros(data_points,1)
		mu = 5
		A = current_value
		for i in range(data_points):
			time = i*time_step #ms
			if time<400:
				webb_input[i]=0.5/0.5*A*math.sin(10*math.pi/1000*time)+mu
			if 400<=time<500:
				webb_input[i]=mu
			if 500<=time<1000:
				webb_input[i]=0.25/0.5*A*math.sin(4*math.pi/1000*time)+mu
			if 1000<=time<1500:
				webb_input[i]=mu
			if 1500<=time<2000:
				webb_input[i]=0.4/0.5*A*math.sin(2*math.pi/1000*time)+mu
			if 2000<=time<2500:
				webb_input[i]=0.1/0.5*A*math.sin(8*math.pi/1000*time)+mu
		return webb_input	
	
	if (type == "webb_train"):
		webb_input = torch.zeros(data_points,1)
		mu = 5
		A = current_value
		for i in range(data_points):
			time = i*time_step #ms
			if time<500:
				webb_input[i]=0.35/0.5*A*math.sin(4*math.pi/1000*time)+mu
			if 500<=time<900:
				webb_input[i]=0.35/0.5*A*math.sin(10*math.pi/1000*time)+mu
			if 900<=time<1000:
				webb_input[i]=mu
			if 1000<=time<1500:
				webb_input[i]=0.5/0.5*A*math.sin(4*math.pi/1000*time)+mu
			if 1500<=time<2000:
				webb_input[i]=mu
			if 2000<=time<2500:
				webb_input[i]=0.5/0.5*A*math.sin(8*math.pi/1000*time)+mu
		return webb_input	
	
	if (type == "webb_advance"):
		webb_input = torch.zeros(data_points,input_neurons)
		mu = 5
		A = current_value
		for i in range(data_points):
			time = i*time_step #s
			if time<450:
				webb_input[i,:]=0.5/0.5*A*math.sin(10*math.pi*time)+mu
			if 450<=time<700:
				webb_input[i,:]=mu+A
			if 700<=time<1200:
				time = time-700
				webb_input[i,:i]=0.5/0.5*A*math.cos(math.pi*time)+mu
			if 1200<=time<1500:
				time = time-1000
				webb_input[i,:]=mu
			if 1500<=time<2000:
				time = time-1500
				webb_input[i,:]=0.4/0.5*A*math.sin(2*math.pi*time)+mu
			if 2000<=time<2500:
				time = time-2000
				webb_input[i,:]=0.1/0.5*A*math.sin(-8*math.pi*time)+mu
			if 2500<=time<3125:
				time = time-2500
				webb_input[i,:]=0.3/0.5*A*math.sin(-4*math.pi*time)+mu
			if 3125<=time<3500:
				webb_input[i,:]=mu-0.3/0.5*A
			if 3500<=time<3875:
				time = time-3500
				webb_input[i,:]=-0.3/0.5*A*math.cos(4*math.pi*time)+mu
			if 3875<=time<4100:
				webb_input[i,:]=mu
			if 4100<=time<4500:
				time = time-4100
				webb_input[i,:]=0.2/0.5*A*math.sin(8*math.pi*time)+mu
			if 4100<=time<4600:
				time = time-4100
				webb_input[i,:]=0.2/0.5*A*math.sin(8*math.pi*time)+mu
			if 4600<=time<5000:
				time = time-4600
				webb_input[i,:]=0.5/0.5*A*math.sin(2*math.pi*time)+mu

		return webb_input	
			
	else: print("Incorrect input sequence")

# seq_len = 50000
# time_step = 0.1
# output = input_cur("webb_advance", seq_len,5,time_step)
# output = output.detach().numpy()
# t = np.arange(0,seq_len*time_step, time_step)

# plt.plot(t,output)
# plt.grid()
# plt.show()