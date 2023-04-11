# import numpy as np
# import matplotlib.pyplot as plt


# tau = 0.90
# I =10
# time = 100
# x=2

# x_arr = np.array([])
# for i in range(time):
#     if i%2 == 0:
#         I = 0
#     else: I = 10   
#     x = x*tau + (1-tau)*I
#     x_arr = np.append(x_arr,x)

# t = np.arange(time)

# plt.plot(t,x_arr)
# plt.grid()
# plt.show()

import yaml

with open("config_bp_izh.yaml","r") as f:
    data = yaml.safe_load(f)

print(data["INITIAL_PARAMS_RANDOM"]["a"])