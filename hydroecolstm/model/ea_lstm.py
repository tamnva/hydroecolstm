
from torch import nn
import torch
from hydroecolstm.model.linears import Linears


# LSTM + Linears

class MultiLinear(nn.Module):
    def __init__(self, input_size_1, input_size_2, output_size, activation_function):
        super(MultiLinear, self).__init__()
        self.Linear_1 = nn.Linear(input_size_1, output_size)
        self.Linear_2 = nn.Linear(input_size_2, output_size)
        self.activation_function = activation_function
        
    def forward(self, x_1, x_2):
        output = self.activation_function(self.Linear_1(x_1) + self.Linear_2(x_2))
        return output

class Ea_Lstm_Linears(nn.Module):
    def __init__(self, config):
        
        super(Ea_Lstm_Linears, self).__init__()
                
        self.static_size = len(config["input_static_features"])
        self.dynamic_size = len(config["input_dynamic_features"])
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.output_size = len(config["target_features"])
        self.linears_num_layers = config["Regression"]["num_layers"]
        self.linears_activation_function = config["Regression"]["activation_function"]
        self.linears_num_neurons = self.find_num_neurons(config=config) 

        # Model structure
        self.i = nn.Sequential(nn.Linear(self.static_size, self.hidden_size), nn.Sigmoid())  
        self.f = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Sigmoid())
        self.g = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Tanh())
        self.o = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Sigmoid())
        self.linear = Linears(num_layers=self.linears_num_layers, 
                              activation_function=self.linears_activation_function,
                              num_neurons=self.linears_num_neurons)
        
    # TODO: This forward function takes too much times, need to improve
    def forward(self, x):
        # Initial hidden, cell states
        c_t = torch.randn(self.hidden_size).unsqueeze(0)
        h_t = torch.randn(self.hidden_size).unsqueeze(0)

        # Forward run
        output = {}
        for key in x.keys(): 
            output[key] = torch.zeros(size=(x[key].shape[0],self.output_size))
            for i in range(x[key].shape[0]):
                i_t = self.i(x[key][i:i+1,self.dynamic_size:])
                f_t = self.f(x[key][i:i+1,:self.dynamic_size], h_t)
                g_t = self.g(x[key][i:i+1,:self.dynamic_size], h_t)             
                o_t = self.o(x[key][i:i+1,:self.dynamic_size], h_t)
                
                c_t = f_t*c_t + i_t*g_t
                h_t = o_t*torch.tanh(c_t)
                
                output[key][i,:] = self.linear(h_t)

        return output
    
    # Find number of neuron in each linear layers, including the input layer
    def find_num_neurons(self, config) -> int:
        # First number of neurons from the input layers ()
        num_neurons = [self.hidden_size]

        if "Regression" in config:
            if len(config["Regression"]["num_neurons"]) > 1:
                for i in range(len(config["Regression"]["num_neurons"])-1):
                    num_neurons.append(config["Regression"]["num_neurons"][i])
        num_neurons.append(self.output_size)

        return num_neurons