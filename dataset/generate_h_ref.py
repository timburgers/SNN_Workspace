import matplotlib.pyplot as plt
import numpy as np
import math
import random

number_of_files= 500
time_step = 0.01		# The sample time per time step [s]
sim_time = 20			# Total length of simulation [s]

for idx in range(number_of_files):
	new_ref_freq = random.randint(4,6)	# per seconds
	minimal_height_change = random.randint(1,4)/10

	z_ref = np.array([])
	time =np.arange(0,sim_time,time_step)

	for t in time:
		
		if t%new_ref_freq==0: 
			if t == 0:
				new_setpoint = random.uniform(0,10)/10	

			if t != 0:
				while abs(new_setpoint-setpoint) <= minimal_height_change:
					new_setpoint = np.round(random.uniform(0,10),2)/10
			setpoint = new_setpoint
		z_ref = np.append(z_ref,setpoint)

	# plt.plot(time,z_ref)
	# plt.grid()
	# plt.show()


	# def save_data(z_ref):
	z_ref.shape= [len(z_ref),1]
	np.savetxt("Sim_data/height_control_PID/zref_norm_positive/dataset_" + str(idx)+ ".csv", z_ref, delimiter=',', header= "timestep = " + str(time_step)+ ", sim time = "+ str(sim_time)+ ", new_ref_freq = "+ str(new_ref_freq)+ ", minimal_height_change = " + str(minimal_height_change))
