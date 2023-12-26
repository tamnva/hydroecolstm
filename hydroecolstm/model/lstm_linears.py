
from torch import nn
from typing import List
from .linears import Linears

# LSTM + Linears
class Lstm_Linears_Base(nn.Module):
    def __init__(self, input_size_lstm:int, output_size_lstm:int, hidden_size_lstm:int, 
                 num_layers_lstm:int, dropout_lstm:float, num_layers_linears:List(int), 
                 act_function_names_linears:List[str], num_neurons_linears:List(int), 
                 batch_first=True, **kwargs):
        
        super(Lstm_Linears_Base, self).__init__()
        
        self.hidden_size_lstm = hidden_size_lstm

        # Standard LSTM from torch
        self.lstm = nn.LSTM(input_size=input_size_lstm, 
                            hidden_size=hidden_size_lstm, 
                            num_layers=num_layers_lstm,
                            dropout=dropout_lstm, 
                            batch_first=batch_first, 
                            **kwargs)
        
        # Fully-connected layer connect hidden and output
        self.linear = Linears(num_layers=num_layers_linears, 
                              act_function_names=act_function_names_linears,
                              num_neurons=num_neurons_linears)
     
    def forward(self, x):
        # get standard LSTM outputs
        lstm_output, _ = self.lstm(x)

        # get final output 
        linear_output = self.linear(lstm_output)
        
        # pass output to activation function
        output = self.activation_function(linear_output)   
        
        # return output
        return output
        
'''    
# Test code
x = torch.rand(10, num_neurons[0])

from torch import nn
model = nn.LSTM(input_size=5, hidden_size=6, num_layers=1,batch_first=True)

x = torch.rand(10, 5)
test = model(x)
test[0].shape

test[0].view(-1, 6) - test[0]
'''


