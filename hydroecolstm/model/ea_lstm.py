
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
    def __init__(self, static_size, dynamic_size, num_layers,
                 hidden_size, **kwargs):
        
        super(EALSTM, self).__init__()
      
        self.c_0 = torch.randn(hidden_size).unsqueeze(0)
        self.h_0 = torch.randn(hidden_size).unsqueeze(0)
        
        self.static_size = static_size
        self.dynamic_size = dynamic_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Model structure parametery
        # Input gate
        self.i = nn.Sequential(nn.Linear(self.static_size, self.hidden_size), nn.Sigmoid())  
        self.f = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Sigmoid())
        self.g = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Tanh())
        self.o = MultiLinear(self.dynamic_size, self.hidden_size, self.hidden_size, nn.Sigmoid())
        
    def forward(self, x):
        o_t = {}
        c_t = {}
        h_t = {}
        
        for key in x.keys():
            c_t = self.c_0
            h_t = self.h_0
            i in range(x[key].shape[0])
                i_t = self.i(x[key][i:i+1,self.dynamic_size:])
                f_t = self.f(x[key][i:i+1,:self.dynamic_size], self.h_t)
                g_t = self.g(x[key][i:i+1,:self.dynamic_size], self.h_t)
                o_t[key] = self.o(x[key][i:i+1,:self.dynamic_size], self.h_t)
        
                c_t = f_t*self.c_t + i_t*g_t
                h_t = o_t[key]*torch.tanh(c_t)
        
        return o_t[key]
        
        
        
        

static_size=2
dynamic_size=3
num_layers=1
hidden_size=4
model = EALSTM(static_size, dynamic_size, num_layers,hidden_size)
x = torch.randn(5, dynamic_size + static_size)

output, _ = model(x)


rnn = nn.LSTM(2, 3)
input = torch.randn(10, 2)
h0 = torch.randn(1, 3)
c0 = torch.randn(1, 3)
output, (hn, cn) = rnn(input, (h0, c0))
