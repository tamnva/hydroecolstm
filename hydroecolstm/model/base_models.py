
from torch import nn

# List of activation function
activation_functions = {"ReLu": nn.ReLU(), "Sigmoid": nn.Sigmoid(), 
                        "Tanh": nn.Tanh(), "Softplus": nn.Softplus(), 
                        "Identity": nn.Identity()}

# LSTM + Linear output layer
class LSTM_DL_Base(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, 
                 dropout, activation_function_name, **kwargs):
        
        super(LSTM_DL_Base, self).__init__()
        
        self.hidden_size = hidden_size
        self.activation_function = activation_functions[activation_function_name]

        # Standard LSTM from torch
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            dropout=dropout, 
                            batch_first=True, 
                            **kwargs)
        
        # Fully-connected layer connect hidden and output
        self.linear = nn.Linear(in_features=hidden_size, 
                                out_features=output_size)
    
    
    def forward(self, x):
        # get standard LSTM outputs
        lstm_output, _ = self.lstm(x)
        # shape output to be (batch_size*seq_length, hidden_size)
        lstm_output = lstm_output.view(-1, self.hidden_size)
        
        # get final output 
        linear_output = self.linear(lstm_output)
        
        # pass output to activation function
        output = self.activation_function(linear_output)   
        
        # return output
        return output

'''
mod = LSTM_DL_Base(input_size=6, output_size=2, hidden_size=30, num_layers=1, 
             dropout=0.0, activation_function_name="Identity")

test = nn.LSTM(input_size=6, 
                    hidden_size=30, 
                    num_layers=1,
                    dropout=0.0, 
                    batch_first=True)

import torch
inputx = x_train_scale['2009'].to(torch.float32)

test(x_train_scale['2009'].to(torch.float32))
mod(x=x_train_scale['2009'])
'''