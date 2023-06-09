import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
import time
import datetime
import random
from scipy.stats import truncnorm
import pickle
import torch

# var = np.array([1,2,3,4,0,0,0,5,6,7,8,0,0,0,9,10,0,0,0,10])


# pos_start = np.array([])
# pos_end = np.array([])

# val=-1

# positive = np.where(var!=0)[0]
# for idx in range(len(positive)):
#     if idx == 0:
#         pos_start = np.append(pos_start,idx)
#     elif positive[idx]-val != 1:
#         pos_start = np.append(pos_start,positive[idx])
#         pos_end = np.append(pos_end,positive[idx-1])
#     if idx ==len(positive)-1:
#         pos_end = np.append(pos_end,positive[idx])
#     val = positive[idx]
    
# filename = "306-soft-bird"
# pickle_in = open("Results_EA/Evotorch/testruns/" + filename+".pkl","rb")
# solution = pickle.load(pickle_in)
# solution = solution.values.detach().numpy()
# # user input
# lower_bound = 0
# upper_bound = 0.2
# range = upper_bound-lower_bound
# my_mean = 0.01
# my_std = 1*(upper_bound-lower_bound)/15
# print(0%3)
# a, b = (lower_bound - my_mean) / my_std, (upper_bound - my_mean) / my_std
# x_range = np.linspace(lower_bound-0.2*range,upper_bound+0.2*range,1000)
# plt.plot(x_range, truncnorm.pdf(x_range, a, b, loc = my_mean, scale = my_std))

# r = truncnorm.rvs(a,b, loc = my_mean, scale = my_std, size=10)
# print(r)
# plt.show()


# print([1]*111)
neurons_l0 = 6
neurons_l1 = 2
a_or = torch.tensor([[1,2,3],[4,5,6]])
a_flat = torch.flatten(a_or)
a = torch.stack((a_flat,-a_flat),dim=1)
a = torch.flatten(a)
a = a.reshape((neurons_l1,neurons_l0))

print(a_or)
print(a)
# print (a)
# print(b)

# cat = torch.stack((a,b),dim=1)
# flat = torch.flatten(cat)
# print(cat)
# print(flat)
# print(flat.shape)
# weight = torch.flatten(torch.cat((self.ff.weight,-1*self.ff.weight),dim=1)).unsqueeze(0)

a 