import numpy as np
import matplotlib.pyplot as plt
import math
import random

for loop in range(100):
    ###############################################################
    mu = 0
    time = 40
    dt = 0.002

    # At zero crossing
    prob_aplitude_change = 0.4 # probability that the amplitude changes to zero crossing
    prob_constant_line_zero = 0.2

    # At top or bottom crossing (both can not be larger than 1!)
    prob_freq_change = 0.3
    prob_constant_line_top_bottom = 0.5

    ROOT_DIR = "Sim_data/derivative/dt0.002_norm_neg"

    ##############################################################################
    # initialize
    A = 0.2
    t = 0
    phase =0
    freq = 1/20
    change_freq = True      
    const_line = False
    signal = np.array([])
    diff_signal = 0.


    for i in range(int(time/dt)):
        t = t + dt
        
        # Select if line should be constant (constant_line = True) or oscilating (constant_line = False)
        if const_line == True:
            current_value = signal[i-1]-mu

            # Check if end of constant is reached
            if t >= const_time:
                const_line = False

                # Set the start phase, depending on where the oscilation stopped
                if current_value ==A:  phase = 0.5*math.pi
                if current_value ==0:  phase = math.pi
                if current_value ==-A: phase = 1.5*math.pi
                t = 0 
        if const_line == False: 
            current_value = A*math.sin(2*math.pi*freq*t+phase)


        # Add signal to list
        signal = np.append(signal,current_value+ mu)


        # Check if line is in top or bottom position (while oscilating, thus const_line = False)
        if (current_value == A or current_value ==-A) and const_line == False:
            rand_num = random.random()

            if prob_constant_line_top_bottom + prob_freq_change >1.0:
                print("Probability of freq change and constant line can not be larger than 1")
                exit()

            if rand_num <= prob_constant_line_top_bottom:
                const_line = True
                const_time = random.randint(2,4)
                t = 0 

                if current_value ==A:  phase = 0.5*math.pi
                if current_value ==-A: phase = 1.5*math.pi

            if rand_num>= (1-prob_freq_change): 
                freq = random.randint(1,4)/20
                t = 0 
                
                if current_value ==A:  phase = 0.5*math.pi
                if current_value ==-A: phase = 1.5*math.pi

        # Change the amplitude of the signal if it is passing through zero (rounded)
        if (round(current_value,12) ==0 and const_line == False):
            random_numb = random.random()

            if prob_aplitude_change + prob_constant_line_zero >1.0:
                print("Probability of freq change and constant line can not be larger than 1")
                exit()

            if random_numb <=prob_aplitude_change:
                A = random.randint(1,5)/10
                t = 0 
                if signal[i-1]-signal[i-2] > 0:
                    phase = 0
                if signal[i-1]-signal[i-2] < 0:
                    phase = math.pi

            # if random_numb >= (1-prob_constant_line_zero): 
            #     const_line = True
            #     const_time = random.randint(3,5)
            #     t = 0
            #     if signal[i-1]-signal[i-2] > 0:
            #         phase = 0
            #     if signal[i-1]-signal[i-2] < 0:
            #         phase = math.pi
        # if i != 0:
        #     diff_signal = (signal[i]-signal[i-1]) /dt
        #     d_signal = np.append(d_signal,diff_signal)


    t = np.arange(0,time,dt)

    diff_signal = np.diff(signal)/dt
    t = t[1:]
    signal = signal[1:]
    t.shape= [len(t),1]
    signal.shape= [len(signal),1]
    diff_signal.shape= [len(diff_signal),1]
    diff_pos_signal = diff_signal.clip(min=0)
    diff_neg_signal = diff_signal.clip(max=0)

    # np.savetxt(ROOT_DIR + "/dataset_"+ str(loop) + ".csv",np.concatenate([t,signal,diff_signal, diff_pos_signal, diff_neg_signal],axis=1),delimiter=',', header="Time (s), Input signal, Derivative, Derivative positive, Derivative negative")


plt.plot(t,signal)
# plt.plot(t, diff_signal)
# plt.scatter(t,signal,color = 'r')
plt.grid()
plt.show()
