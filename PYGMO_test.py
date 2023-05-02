import pygmo as pg
import torch
import pygad.torchga as torchga
from SNN_Izh_LI_init import Izhikevich_SNN, initialize_parameters
from Izh_LI_EA_PYGAD import get_dataset
import yaml
import numpy as np
from tqdm import trange
import multiprocessing as mp

class izh_EA_pygmo:
    def __init__(self):
        ### Read config file
        with open("config_PYGMO.yaml","r") as f:
            self.config = yaml.safe_load(f)

        self.param_init = initialize_parameters(self.config)
        _device = "cpu"
        self.SNN_izhik = Izhikevich_SNN(self.param_init, _device, self.config)
        self.input_data, self.target_data = get_dataset(self.config, self.config["DATASET_NUMBER"], self.config["SIM_TIME"])
        self.loss_function = torch.nn.MSELoss()

    def fitness(self, chromosome):
        #### Initialize neuron states (u, v, s) 
        snn_states = torch.zeros(3, 1, self.SNN_izhik.neurons)
        snn_states[1,:,:] = -70			#initialize V
        snn_states[0,:,:] = -20 		#initialize U
        LI_state = torch.zeros(1,1)


        izh_output, izh_state, predictions = torchga.predict(self.SNN_izhik,
                                            chromosome,
                                            self.input_data,
                                            snn_states, #(u, v, s) 
                                            LI_state) #data in form: input, state_snn, state LI

        predictions = predictions[:,0,0]
        solution_fitness = self.loss_function(predictions, self.target_data).detach().numpy()
        print(solution_fitness)
        return [solution_fitness]
    
    def batch_fitness(self, chromosomes):
        len_single_chromosome = 111
        dpvs = chromosomes.reshape(len(chromosomes)//len_single_chromosome, len_single_chromosome)

        inputs, fitnesses = [], []
        for dpv in dpvs:
            inputs.append([list(dpv)])

        # cpu_count = len(os.sched_getaffinity(0))
        cpu_count = mp.cpu_count()
        with mp.get_context("spawn").Pool(processes=int(cpu_count-4)) as pool:
            outputs = pool.map(self.fitness, inputs)

        for output in outputs:
            fitnesses.append(output)

        return fitnesses

    def get_bounds(self):
        lower_bounds = []
        upper_bounds = []
        ### Get the structure and order of the genome
        param_model =torchga.model_weights_as_dict(self.SNN_izhik, np.ones(111)) # fill with dummy inputs

        ### Check if there is a lim in the config and otherwise add None to it
        for name, value in param_model.items():
            number_of_params = len(torch.flatten(value).detach().numpy())
            for iteration in range(number_of_params):
                if name in self.config["PARAMETER_BOUNDS"]:

                    # Check if parameters is the weights
                    if name.split(".")[-1] == "weight":

                        # Set the first 5 neurons to a positive bound 
                        if iteration<number_of_params/2:
                            lower_bounds.append(0)
                            upper_bounds.append(self.config["PARAMETER_BOUNDS"][name]["high"])

                        # Set the last 5 neurons the the negative bound
                        else:
                            lower_bounds.append(self.config["PARAMETER_BOUNDS"][name]["low"])
                            upper_bounds.append(0)
                    else:
                        lower_bounds.append(self.config["PARAMETER_BOUNDS"][name]["low"])
                        upper_bounds.append(self.config["PARAMETER_BOUNDS"][name]["high"])
                else:
                    lower_bounds.append(None)
                    upper_bounds.append(None)
    
        return (lower_bounds , upper_bounds)

    def get_name(self):
        return "Sphere Function"

    def get_extra_info(self):
        return "\tDimensions: " + str(111)







if __name__ == "__main__":
    bfe = True

    prob = pg.problem(izh_EA_pygmo())

    individuals_list = []
    fitness_list = []

    uda = pg.cmaes(gen = 1, sigma0 = 0.5, memory = True)
    # uda.set_bfe(pg.bfe())
    algo = pg.algorithm(uda)

    algo.set_verbosity(1)

    pop = pg.population(prob, prob.batch_fitness(), 10)
    # print(pop.get_x()[pop.best_idx()])
    # pop = algo.evolve(pop)
    # print(pop.get_x()[pop.best_idx()]) 
    for i in trange(10):      
        pop = algo.evolve(pop)
        individuals_list.append(pop.get_x()[pop.best_idx()])
        fitness_list.append(pop.get_f()[pop.best_idx()])

    print(pop.champion_f) 