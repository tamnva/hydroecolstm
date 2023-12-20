import torch
from hydroecolstm.utility.loss_function import LossFunction
from hydroecolstm.model.base_models import LSTM_DL_Base

class LSTM_DL:
    # Initialization
    def __init__(self, config:dict, **kwargs): 
        self.learning_rate = config["learning_rate"]
        self.n_epochs = config["n_epochs"]
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        self.input_size = self.get_input_size(config)
        self.output_size = len(config["target_features"])
        self.learning_rate = config["learning_rate"]
        self.n_epochs = config["n_epochs"]
        self.hidden_size = config["hidden_size"]
        self.dropout = config["dropout"]*min(1.0, self.num_layers - 1.0)
        self.warmup_length = config["warmup_length"]
        self.activation_function_name = config["activation_function_name"]
        self.objective_function_name = config["objective_function_name"]
        self.loss_function = LossFunction()
        self.model = LSTM_DL_Base(input_size=self.input_size, 
                                  output_size=self.output_size,
                                  hidden_size=self.hidden_size, 
                                  num_layers=self.num_layers,
                                  dropout=self.dropout, 
                                  activation_function_name=self.activation_function_name,
                                  **kwargs)
        
    # Train the model
    def train(self, x_train: dict[str: torch.Tensor], y_train: dict[str: torch.Tensor]):
        
        # Optimization function
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Train the model
        for epoch in range(self.n_epochs):
            
            # Save model output in dict[str:torch.Tensor]
            y_predict = {}
            # Loop over object_id
            for key in x_train.keys():
                y_predict[key] = self.model(x_train[key])

            # Reset the gradients to zero
            self.optim.zero_grad()
            
            # Caculate loss function
            error, error_avg = self.loss_function(y_true=y_train, 
                                                  y_predict=y_predict,
                                                  nskip=self.warmup_length,
                                                  objective_function_name=self.objective_function_name)
          
            # Error back propagation LSTM.state_dict()
            error_avg.backward()
            # Update weights and biases
            self.optim.step()
            # Print to screen the process  LSTM.state_dict()
            #if epoch % 2 == 0: 
            print(f"Epoch [{epoch+1}/{self.n_epochs}], err: {error_avg:.8f}")
            
        return self.model, y_predict  #, error, error_avg
    
    # testing function
    def forward(self, x):
        # Save model output in dict[str:torch.Tensor]
        y_predict = {}
        
        # Loop over object_id
        for key in x.keys():
            y_predict[key] = self.model(x[key])
            
        return y_predict
    
    # get input size
    def get_input_size(self, config) -> int:
        if "input_static_features" in config:
            input_size = (len(config["input_dynamic_features"]) + 
                          len(config["input_static_features"]))
        else:
            input_size = len(config["input_dynamic_features"])
        return input_size
    

