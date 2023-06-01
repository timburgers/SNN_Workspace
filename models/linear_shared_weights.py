import torch
import torch.nn as nn
from spiking.torch.neurons.lif import BaseLIF, SoftLIF


class SHARED_BaseLinear(nn.Module):
    """
    Base densely-connected feedforward linear layer with:
    - no bias
    """

    neuron_model = None

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn, bias, *time_step):
        super().__init__()

        self.ff = nn.Linear(input_size, output_size, bias=bias)
        self.neuron = self.neuron_model(fixed_params, learnable_params, spike_fn, *time_step)

    def forward(self, state, input_):
        weight = torch.flatten(torch.cat((self.ff.weight,-1*self.ff.weight),dim=1)).unsqueeze(0) #shape (neurons,1) (while self.ff.weights is of shape (neurons/2))
        
        ff = torch.nn.functional.linear(input_,weight)
        state, output = self.neuron(state, ff)

        return state, output


class SHARED_BaseRecurrentLinear(nn.Module):
    """
    Base densely-connected recurrent linear layer with:
    - no bias
    """

    neuron_model = None

    def __init__(self, input_size, output_size, fixed_params, learnable_params, spike_fn):
        super().__init__()

        self.ff = nn.Linear(input_size, output_size, bias=True)
        self.rec = nn.Linear(output_size, output_size, bias=False)
        self.neuron = self.neuron_model(fixed_params, learnable_params, spike_fn)

    def forward(self, state, input_):
        weight = torch.flatten(torch.cat((self.ff.weight,-1*self.ff.weight),dim=1)).unsqueeze(1) #shape (neurons,1) (while self.ff.weights is of shape (neurons/2))
        bias = torch.flatten(torch.cat((self.ff.bias,self.ff.bias),dim=1))
        
        ff =torch.nn.functional.linear(input_,weight,bias)


        state = state if state is not None else self.neuron.reset_state(ff)

        s = self.neuron.get_spikes(state)
        rec = self.rec(s)

        state, output = self.neuron.activation(state, ff + rec)  # .activation to save a state check
        return state, output


class SHARED_LinearLIF(SHARED_BaseLinear):
    """
    Linear layer with LIF activation.
    """

    neuron_model = BaseLIF


class SHARED_RecurrentLinearLIF(SHARED_BaseRecurrentLinear):
    """
    Recurrent linear layer with LIF activation.
    """

    neuron_model = SoftLIF


