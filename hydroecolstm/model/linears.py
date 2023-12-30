
from torch import nn
from typing import List

class Linears(nn.Module):
    '''
    Multi-linear layers with different activation functions

    Parameters
    ----------
    num_layers : int
        Number of linear layers.
    activation_function: List[str]
        Name of  activation function for each linear layer, available function 
        names are "ReLu", "Sigmoid", "Tanh", "Softplus", "Identity"
    num_neurons: List[int]
        Number of neurons in each layers, including from the input size to the
        the multi-linear layers. Therefore, the len(num_neurons) = num_layers + 1

    Attributes
    ----------
    model : nn.Module
        The multi-linear layers
        
    Examples
    --------
    >>> import torch
    >>>
    >>> act_function_names = ["Identity", "Identity", "Identity"]
    >>> num_layers = 3
    >>> num_neurons = [2,3,4,3]
    >>>
    >>> x = torch.rand(10, num_neurons[0])
    >>> model = Linears(num_layers, act_function_names, num_neurons)
    >>>
    >>> # Forward run
    >>> model(x)
    '''
        
    def __init__(self, num_layers: int, activation_function: List[str], 
                 num_neurons: List[int]):
        super(Linears, self).__init__()
        
        # Activation functions
        activation_functions = {"ReLu": nn.ReLU(), "Sigmoid": nn.Sigmoid(), 
                                "Tanh": nn.Tanh(), "Softplus": nn.Softplus(), 
                                "Identity": nn.Identity()}
        
        # Create list to store different linear layers
        layers = []
        
        # Create layers of user-defined network
        for i in range(num_layers):
            layers.append(nn.Linear(num_neurons[i], num_neurons[i+1]))
            layers.append(activation_functions[activation_function[0]])
        
        # Combined all layers together using sequential
        self.model = nn.Sequential(*layers)
        
    # Forward run
    def forward(self, x):
        # model output
        output = self.model(x)
        
        # Return output
        return output
    
    