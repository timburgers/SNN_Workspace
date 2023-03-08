import torch
import torch.nn as nn
from spiking.torch.layers.linear import LinearLIF
from spiking.torch.utils.surrogates import get_spike_fn

class ProportionalNeuron(torch.nn.Module):
    def __init__(self, input_groups:int, neuron_params=None):
        super(ProportionalNeuron,self).__init__()

        self.input_features = input_groups*2
        self.hidden_features = self.input_features

        if neuron_params == None:
            v_thresh = 1.0
            tau_syn = 0.08
            tau_mem = 0.03
        else:
            v_thresh = neuron_params["v_thresh"]
            tau_syn = neuron_params["tau_syn"]
            tau_mem = neuron_params["tau_mem"]

        leak_syn = torch.ones(self.hidden_features)*(1-tau_syn) 
        leak_mem = torch.ones(self.hidden_features)*(1-tau_mem)
        v_thresh =  torch.ones(self.hidden_features)*v_thresh

        params_fixed = dict(
            thresh=v_thresh
        )

        params_learnable = dict(
            leak_i= leak_syn,
            leak_v = leak_mem
        )
        # first paramaters are the height and slope of the arctan function (i think so)
        self.l1 = LinearLIF(self.input_features,self.hidden_features,params_fixed,params_learnable,get_spike_fn("ArcTan", 1.0, 20.0))


        # Tot aan hier om gewoon een neuron proportioneel te laten spiken
        weight = 0.4
        weight_list = []

        for i in range(input_groups):
            weight_list.append(torch.tensor([[1,0],[0,1]]*weight))

        new_weights = torch.block_diag(*weight_list)
        self.l1.ff.weight = torch.nn.parameter.Parameter(new_weights)



    def reset(self,batch_size=1):
        """
        Reset the states
        
        :param batch_size: desired batch_size
        :returns: A tensor containing the states all set to zero
        
        """
        fake_input = torch.zeros(
            [1,batch_size,self.input_features]
        )
        _, single_step = self.forward(fake_input)
        return single_step



    def forward(self,x, states=None, record=None):
        seq_length, batch_size, n_inputs = x.size()

        if states==None:
            s1 = None
        else:
            s1 = states
        
        voltages = []

        if record:
            self.recording = []

        for ts in range(seq_length):
            z = x[ts,:,:]
            s1,z = self.l1(s1,z)

            if record:
                self.recording.append(s1[2])
            voltages += [z]

        output = torch.stack(voltages)

        return output, s1

p_neurons=ProportionalNeuron(1)