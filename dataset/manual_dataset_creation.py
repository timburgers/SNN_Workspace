import matplotlib.pyplot as plt 
import numpy as np
import math

sim_time = 13
time_step = 0.002

file_name = "manual_dataset"



man_input = np.zeros(int(sim_time/time_step))
mu = 0
A = 0.5
for i in range(len(man_input)):
    time = i*time_step #ms
    if time<3:
        t=time
        man_input[i]=A/2*math.sin(1/6*2*math.pi*t+3/2*math.pi)+A/2
    if 3<=time<6:
        t=time-3
        man_input[i]=A*math.sin(1/6*2*math.pi*t+1/2*math.pi)+mu
    if 6<=time<7.5:
        man_input[i]=-A
    if 7.5<=time<9:
        t=time-7.5
        man_input[i]=A*math.sin(1/6*2*math.pi*t+3/2*math.pi)+mu
    if 9<=time<13:
        t=time-9
        man_input[i]=A/0.5*0.3*math.sin(1/2*2*math.pi*t)+mu


time = np.arange(0, sim_time, time_step)
plt.plot(time,man_input)
plt.grid()
plt.show()


diff_signal = np.diff(man_input)/time_step
# prepend the first input of the diff list (such that input and diff are of the same length)
diff_signal = np.insert(diff_signal,0,diff_signal[0])

plt.plot(time,diff_signal)
plt.grid()
plt.show()

### Reshape arrays for csv saving
diff_signal.shape   = [len(diff_signal),1]
man_input.shape     =[len(man_input),1]
time.shape          =[len(time),1]
diff_pos_signal = diff_signal.clip(min=0)
diff_neg_signal = diff_signal.clip(max=0)



np.savetxt(file_name + ".csv",np.concatenate([time,man_input,diff_signal,diff_pos_signal,diff_neg_signal],axis=1),delimiter=',', header="Time (s), Input signal, Derivative, Pos Derivative, Neg Derivative")
