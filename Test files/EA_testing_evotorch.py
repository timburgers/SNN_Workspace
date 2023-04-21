from evotorch import Problem
from evotorch.algorithms import SNES
from evotorch.logging import StdOutLogger, PicklingLogger,PandasLogger
import torch
import pickle


# Define a function to minimize
def sphere(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.pow(2.0))


# Define a Problem instance wrapping the function
# Solutions have length 10
problem = Problem("min", sphere, solution_length=5, initial_bounds=(-1, 1))

# Instantiate a searcher
searcher = SNES(problem, stdev_init=5)

# Create a logger
logger = PicklingLogger(searcher,interval=50)


# Evolve!
searcher.run(200)
print(logger.unpickle_last_file())

