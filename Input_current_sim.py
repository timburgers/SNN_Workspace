import random
import torch

def input_cur(type, length, current_value): 
	"""
	Type options: spike, constant, triangle & step
	"""

	if (type == "spike"):
		# Generate random spike pattern
		percentage_of_spikes = 0.02
		spike_list = [0]*int(length*(1-percentage_of_spikes))+[0.1]*int(length*percentage_of_spikes)
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
		step_percentage = 10

		step_input = torch.zeros(length,1)
		for i in range(length):
			if i < (length/step_percentage): step_input[i,0]=0
			else: step_input[i,0]=current_value
		return step_input
	
	else: print("Incorrect input sequence")
