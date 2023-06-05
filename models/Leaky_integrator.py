from spiking.torch.neurons.lif import BaseNeuron
from spiking.torch.layers.linear import BaseLinear

import torch

class Leaky_integrator_neuron(BaseNeuron):
    """
    leaky integrator to create a spike trace output
    - optionally learnable parameters; either per-neuron or single
    """

    state_size = 1
    neuron_params = ["leak"]

    def __init__(self, fixed_params, learnable_params, spike_fn, _layer_setting):
        super().__init__(self.state_size, fixed_params, learnable_params)

        # check parameters are there
        for p in ["leak"]:
            assert hasattr(self, p), f"{p} not found in {self}"

        # spike mechanism
        self.spike = spike_fn

    def activation(self, state, input_):
        # unpack state; spikes always last
        v = state

        # get parameters
        leak = self.get_leak()

        # voltage update: leak, reset, integrate
        v = self.update_mem(v, leak, input_)
        
        # return none for output, since it is non spiking
        return v, None

    def get_leak(self):
        return torch.clamp(self.leak, min=0, max=1)
    
    @staticmethod
    def update_mem(v,leak,input):
        return v * leak + (1-leak)*input



class Linear_LI_filter(BaseLinear):
    """
    Linear layer with leaky integrator neuron.
    """

    neuron_model = Leaky_integrator_neuron

