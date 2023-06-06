import matplotlib.pyplot as plt
import numpy as np
import math
import random

time_step = 0.01		# The sample time per time step [s]
sim_time = 30			# Total length of simulation [s]
freq_list = np.arange(2,10,0.1)
time =np.arange(0,sim_time,time_step)


for idx in range(len(freq_list)):
	error_arr = np.array([])
	for t in time:
		error = math.sin(2*math.pi*freq_list[idx]*t)
		error_arr = np.append(error_arr,error)

	d_error = np.diff(error_arr)
	d_error = np.insert(d_error,0,0)
	# plt.plot(time,error_arr,label="error")
	# plt.plot(time,d_error,label = "derivative")
	# plt.legend()
	# plt.grid()
	# # plt.show()

	error_arr.shape=[len(error_arr),1]
	d_error.shape=[len(d_error),1]


	np.savetxt("Sim_data/height_control_PID/sine_derivative_large/dataset_" + str(idx)+ ".csv", np.concatenate([error_arr,d_error],axis=1), delimiter=',', header= "timestep = " + str(time_step)+ ", sim time = "+ str(sim_time)+ ", sine_freq = "+ str(freq_list[idx]))
