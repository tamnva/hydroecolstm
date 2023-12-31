
from torch import nn
import torch


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

class EALSTM(nn.Module):
    def __init__(self, config):
        
        super(EALSTM, self).__init__()
                
        self.static_size = len(config["input_static_features"])
        self.dynamic_size = len(config["input_dynamic_features"])
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.output_size = len(config["target_features"])

        self.c_0 = torch.randn(self.hidden_size).unsqueeze(0)
        self.h_0 = torch.randn(self.hidden_size).unsqueeze(0)

        # Model structure
        self.i = nn.Sequential(nn.Linear(self.static_size, self.hidden_size), nn.Sigmoid())  
        self.f = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Sigmoid())
        self.g = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Tanh())
        self.o = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Sigmoid())
        self.linear = nn.Sequential(nn.Linear(self.hidden_size, self.output_size), nn.Identity())  
        
    def forward(self, x):
        output = {}
       
        for key in x.keys():
            c_t = self.c_0
            h_t = self.h_0
            for i in range(x[key].shape[0]):
                i_t = self.i(x[key][i:i+1,self.dynamic_size:])
                f_t = self.f(x[key][i:i+1,:self.dynamic_size], h_t)
                g_t = self.g(x[key][i:i+1,:self.dynamic_size], h_t)             
                o_t = self.o(x[key][i:i+1,:self.dynamic_size], h_t)
                out = self.linear(o_t)
                
                c_t = f_t*c_t + i_t*g_t
                h_t = o_t*torch.tanh(c_t)
                
                
                if i == 0:
                    output[key] = out
                else:
                    output[key] = torch.cat((output[key], out))
                
        return output