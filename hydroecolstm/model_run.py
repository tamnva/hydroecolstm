#!/usr/bin/env python

from hydroecolstm.data.read_data import read_scale_data
from hydroecolstm.model.create_model import create_model
from hydroecolstm.train.trainer import Trainer

# Function to train and test the model 
def run_config(config):

    # Read and split data
    data = read_scale_data(config)
    
    # Create the model
    model = create_model(config)
        
    # Train with train dataset
    trainer = Trainer(config, model)
    model = trainer.train(data['x_train_scale'], data['y_train_scale'], 
                          data['x_valid_scale'], data['y_valid_scale'])
    
    # Save train loss per epoch and best train loss
    data["loss_epoch"] = trainer.loss_epoch
    
    return model, data

# Run with input with Dict[str:torch.Tensor] and torch.no_grad()
def forward_run(model, data):
    y_train_simulated_scale = model.evaluate(data['x_train_scale'])
    y_valid_simulated_scale = model.evaluate(data['x_valid_scale'])
    y_test_simulated_scale = model.evaluate(data['x_test_scale'])
    
    # Inverse scale/transform back simulated result to real scale
    data["y_train_simulated"] = data['y_scaler'].inverse(y_train_simulated_scale)
    data["y_valid_simulated"] = data['y_scaler'].inverse(y_valid_simulated_scale)
    data["y_test_simulated"] = data['y_scaler'].inverse(y_test_simulated_scale)
    
    return data