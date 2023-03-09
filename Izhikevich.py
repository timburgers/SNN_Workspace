import torch

from spiking.torch.neurons.base import BaseNeuron
from spiking.torch.utils.quantization import quantize
from spiking.torch.layers.linear import BaseLinear



class BaseIzhikevich(BaseNeuron):
    """
    Base LIF neuron with:
    - stateful mem potential and recovery variable
    - hard reset of membrane potential
    - optionally learnable parameters; either per-neuron or single
    """
    # membrane potential (v), recovery variable (u), output spike (s)
    state_size = 3
    neuron_params = ["a","b","c","d" "thresh","time_step"]
    

    def __init__(self, fixed_params, learnable_params, spike_fn):
        super().__init__(self.state_size, fixed_params, learnable_params)

        # check parameters are there
        for p in ["a","b","c","d", "thresh","time_step"]:
            assert hasattr(self, p), f"{p} not found in {self}"

        # spike mechanism for back prop
        self.spike = spike_fn

    def activation(self, state, input_):
        # unpack state; spikes always last
        u, v, s = state

        # get parameters
        # TODO: replace with pre-forward hook?
        a,b,c,d = self.get_param()
        thresh = self.get_thresh()
        dt = self.get_time_step()

        # voltage update + reset + integrate
        v = self.update_mem(v, u, input_, s, c, dt)

        # recovery update + reset
        u = self.update_recov(a, b, d, v, u, s, dt)

        # spike
        s = self.spike(v - thresh)

        return torch.stack([u, v, s]), s

    @staticmethod
    def get_spikes(state):
        _, _, s = state
        return s

    def get_param(self):
        return self.a, self.b, self.c, self.d

    def get_thresh(self):
        return self.thresh
    
    def get_time_step(self):
        return self.time_step

    @staticmethod
    def update_recov(a,b,d,v,u,reset,dt):
        return u + dt*a*(b*v - u) + d*reset
    
    @staticmethod
    def update_mem(v,u,I,reset,c,dt):
        return (v + dt*(0.04*v**2 + 5*v + 140 + - u + I))*(1-reset) + reset*c



class LinearIzhikevich(BaseLinear):
    """
    Linear layer with Izhikevich activation.
    """

    neuron_model = BaseIzhikevich