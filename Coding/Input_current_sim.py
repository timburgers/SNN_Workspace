import random
import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def input_cur(type, length, current_value, time_step): 
	"""
	Type options: spike, constant, triangle, step or webb
	"""

	if (type == "spike"):
		# Generate random spike pattern
		percentage_of_spikes = 0.002
		spike_list = [0]*int(length*(1-percentage_of_spikes))+[current_value]*int(length*percentage_of_spikes)
		random.shuffle(spike_list)
		spike_list = torch.FloatTensor(spike_list)
		spike_input = torch.reshape(spike_list,(length,1))
		return spike_input

	if (type == "constant"):
		# Set constant input current
		const_input = torch.zeros(length,1)
		for i in range(length):
			const_input[i,0]=current_value
		return const_input

	if (type == "triangle"):
		# Set constant input current
		triangle_input = torch.zeros(length,1)
		for i in range(length):
			if i < (length/2): triangle_input[i,0]=current_value*i
			else: triangle_input[i,0]=current_value*(length - i)
		return triangle_input

	if (type == "step"):
		step_percentage = 0.56

		step_input = torch.zeros(length,1)
		for i in range(length):
			if i < (length*step_percentage): step_input[i,0]=5*(10**-9)
			else: step_input[i,0]=20*(10**-9)#current_value
		return step_input
	
	if (type == "webb"):
		webb_input = torch.zeros(length,1)
		mu = 5
		A = current_value
		for i in range(length):
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
			
	else: print("Incorrect input sequence")


