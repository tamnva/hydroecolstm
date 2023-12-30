
from typing import Dict
import torch
from hydroecolstm.utility.loss_function import LossFunction

# LSTM + Linears
class Train():
    def __init__(self, config, model, **kwargs):
        
        super(Train, self).__init__()

        # Training parameters
        self.lr = config["learning_rate"]
        self.objective_function_name = config["objective_function_name"]
        self.n_epochs = config["n_epochs"]
        self.nskip = config["warmup_length"]
        self.loss_function = LossFunction()
        self.model = model
    
    # Train function
    def __call__(self, x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]):
        
        # Optimization function
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                
        # Train the model
        for epoch in range(self.n_epochs):
            
            # Get model output
            y_predict = self.model(x)

            # Reset the gradients to zero
            optim.zero_grad()
            
            # Caculate loss function
            loss, loss_avg =\
                self.loss_function(y_true=y, y_predict=y_predict, nskip=self.nskip,
                                   objective_function_name=self.objective_function_name)
          
            # Error back propagation LSTM.state_dict()
            loss_avg.backward()
            
            # Update weights and biases
            optim.step()
            
            # Print to console
            print(f"Epoch [{epoch+1}/{self.n_epochs}], avg_loss = {loss_avg:.8f}")
            
        return self.model, y_predict
    
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

        if "REG" in config:
            if len(config["REG"]["num_neurons"]) > 1:
                for i in range(len(config["REG"]["num_neurons"])-1):
                    num_neurons.append(config["REG"]["num_neurons"][i])
        num_neurons.append(self.output_size)

        return num_neurons
                    
            