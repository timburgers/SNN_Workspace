#Created on 17/04/2023 at 16:03
import copy
import numpy
import torch


def model_weights_as_vector(model):
    weights_vector = []

    for curr_weights in model.state_dict().values():
        # Calling detach() to remove the computational graph from the layer.
        # cpu() is called for making shore the data is moved from GPU to cpu
        # numpy() is called for converting the tensor into a NumPy array.
        curr_weights = curr_weights.cpu().detach().numpy()
        vector = numpy.reshape(curr_weights, newshape=(curr_weights.size))
        weights_vector.extend(vector)

    return numpy.array(weights_vector)

def model_weights_as_dict(model, weights_vector):
    weights_dict = model.state_dict()

    start = 0
    for key in weights_dict:
        # Calling detach() to remove the computational graph from the layer. 
        # cpu() is called for making shore the data is moved from GPU to cpu
        # numpy() is called for converting the tensor into a NumPy array.
        w_matrix = weights_dict[key].cpu().detach().numpy()
        layer_weights_shape = w_matrix.shape
        layer_weights_size = w_matrix.size

        layer_weights_vector = weights_vector[start:start + layer_weights_size]
        layer_weights_matrix = numpy.reshape(layer_weights_vector, newshape=(layer_weights_shape))
        weights_dict[key] = torch.from_numpy(layer_weights_matrix)

        start = start + layer_weights_size

    return weights_dict

def predict(model, solution, *data):
    # Fetch the parameters of the best solution.
    model_weights_dict = model_weights_as_dict(model=model,
                                               weights_vector=solution)

    # Use the current solution as the model parameters.
    _model = copy.deepcopy(model)
    _model.load_state_dict(model_weights_dict)

    izh_output, izh_state, predictions = _model(*data)

    return predictions

class TorchGA:

    def __init__(self, model, num_solutions,config):

        """
        Creates an instance of the TorchGA class to build a population of model parameters.

        model: A PyTorch model class.
        num_solutions: Number of solutions in the population. Each solution has different model parameters.
        """
        
        self.model = model

        self.num_solutions = num_solutions

        self.config = config

        # A list holding references to all the solutions (i.e. networks) used in the population.
        self.population_weights = self.create_population()

        

    def create_population(self):

        """
        Creates the initial population of the genetic algorithm as a list of networks' weights (i.e. solutions). Each element in the list holds a different weights of the PyTorch model.

        The method returns a list holding the weights of all solutions.
        """

        model_weights_vector = model_weights_as_vector(model=self.model)

        net_population_weights = []
        net_population_weights.append(model_weights_vector)

        for idx in range(self.num_solutions-1):


            net_weights = init_param_in_torchga_create_pop(self.config)

            # Appending the weights to the population.
            net_population_weights.append(net_weights)

        return net_population_weights


### New self made function to add the manual initialization using the values in the config file
def init_param_in_torchga_create_pop(config):
	init_param = numpy.array([])
	neurons = config["NEURONS"]
	# print("Initialization is set to : ", config["INIT_SETTING"])
	if config["INIT_SETTING"] == "random":
		config_rand = config["INITIAL_PARAMS_RANDOM"]
		w1 = numpy.random.uniform(config_rand["weights_1"][0],config_rand["weights_1"][1],size=(neurons))
		a = numpy.random.uniform(config_rand["a"][0],config_rand["a"][1], size=(neurons))
		b= numpy.random.uniform(config_rand["b"][0],config_rand["b"][1], size=(neurons))
		c = numpy.random.uniform(config_rand["c"][0],config_rand["c"][1], size=(neurons))
		d = numpy.random.uniform(config_rand["d"][0],config_rand["d"][1], size=(neurons))
		v2 = numpy.random.uniform(config_rand["v2"][0],config_rand["v2"][1], size=(neurons))
		v1 = numpy.random.uniform(config_rand["v1"][0],config_rand["v1"][1], size=(neurons))
		v0 = numpy.random.uniform(config_rand["v0"][0],config_rand["v0"][1], size=(neurons))
		tau_u = numpy.random.uniform(config_rand["tau_u"][0],config_rand["tau_u"][1], size=(neurons))
		thres = numpy.random.uniform(config_rand["threshold"][0],config_rand["threshold"][1], size=(neurons))
		w2 = numpy.random.uniform(config_rand["weights_2"][0],config_rand["weights_2"][1],size=(neurons))	# random weights [-1,1]
		leak = numpy.random.uniform(config_rand["leak"][0],config_rand["leak"][1], size=(1))
		
		# Set half of the initial input and corresponding output weights to negative values
		if config["HALF_NEGATIVE_WEIGHTS"] == True:
			for idx in range(int(neurons/2)):
				w1[idx] = w1[idx] *-1
				w2[idx] = w2[idx]*-1

		init_param = numpy.concatenate((w1,a,b,c,d,v2,v1,v0,tau_u,thres,w2,leak), axis=None)

	else: 
		print("Init setting not found")
		exit()
	return init_param
