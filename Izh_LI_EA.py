from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PicklingLogger,PandasLogger
from dataset_creation.pytorch_dataset import Dataset_derivative
from SNN_Izh_LI_init import Izhikevich_SNN, initialize_parameters
from wandb_log_functions import *
import torch
import pickle
import yaml
import wandb

### Read config file
with open("config_Izh_LI_EA.yaml","r") as f:
    config = yaml.safe_load(f)
device = "cpu"

### Initialize SNN + Dataset
param_init = initialize_parameters(config)
SNN_izhik = Izhikevich_SNN(param_init, device, config).to(device)
dataset = Dataset_derivative(config)

### Set weight & biases
run = wandb.init(project= "SNN_Izhikevich_P", mode=config["WANDB_MODE"], reinit=True, config=config)	
set_config_wandb(SNN_izhik, config)
wandb.watch(SNN_izhik, log="parameters", log_freq=1)


