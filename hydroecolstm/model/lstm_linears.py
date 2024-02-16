
from torch import nn
import torch
from hydroecolstm.model.linears import Linears

# LSTM + Linears
class Lstm_Linears(nn.Module):
    def __init__(self, config, **kwargs):
        
        super(Lstm_Linears, self).__init__()

        # Model structure parametery
        self.input_size = self.get_input_size(config)
        self.output_size = len(config["target_features"])
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]*min(1.0, self.num_layers - 1.0)
        self.linears_num_layers = config["Regression"]["num_layers"]
        self.linears_activation_function = config["Regression"]["activation_function"]
        self.linears_num_neurons = self.find_num_neurons(config=config) 
        
        # Standard LSTM from torch
        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            **kwargs)


        # Fully-connected layer connect hidden and output
        self.linear = Linears(num_layers=self.linears_num_layers, 
                              activation_function=self.linears_activation_function,
                              num_neurons=self.linears_num_neurons)
     
    # Forward mode run with torch.Tensor as usrual
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        # get standard LSTM outputs
        y_predict, _ = self.lstm(x)
        
        # get final output 
        y_predict = self.linear(y_predict)  
        
        # return output
        return y_predict

    # In the evaluate mode run with Dict[str: tensor]
    def evaluate(self, x:dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        
        with torch.no_grad():
            y_predict = {}
            
            for key in x.keys():
                # get standard LSTM outputs
                y_predict[key], _ = self.lstm(x[key])
                
                # get final output 
                y_predict[key] = self.linear(y_predict[key])  
        
        # return output
        return y_predict

    
    # get input size
    def get_input_size(self, config) -> int:
        if "input_static_features" in config:
            input_size = (len(config["input_dynamic_features"]) + 
                          len(config["input_static_features"]))
        else:
            input_size = len(config["input_dynamic_features"])
        return input_size
    
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