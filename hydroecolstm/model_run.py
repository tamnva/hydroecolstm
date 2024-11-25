#!/usr/bin/env python

from hydroecolstm.data.read_data import read_scale_data
from hydroecolstm.model.create_model import create_model
from hydroecolstm.train.trainer import Trainer
from ray import tune, air
from pathlib import Path
from datetime import datetime
import copy
import os
import torch

# Function to train and test the model 
def run_config(config):

    # Read and split data
    data = read_scale_data(config)
    
    # Convert config to search space
    search_space = config_to_search_space(config, 
                                          data['x_train_scale'], 
                                          data['y_train_scale'],
                                          data['x_valid_scale'], 
                                          data['y_valid_scale'])
    
    if search_space["is_manual_optim"]:
        
        # Create the model
        model = create_model(config)
        
        # Train with train dataset
        trainer = Trainer(config, model)
        model = trainer.train(data['x_train_scale'], data['y_train_scale'],
                              data['x_valid_scale'], data['y_valid_scale'])
        
        # Save train loss per epoch and best train loss
        data["loss_epoch"] = trainer.loss_epoch
        data["best_train_loss"] = trainer.best_train_loss
        best_config = config
        
    else:
        
        tuner = tune.Tuner(loss,
                           param_space=search_space,
                           run_config=air.RunConfig(
                               Path(config["output_directory"][0],
                                    datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                    )
                               )
                           )
        result = tuner.fit()
        
        # Create model 
        best_result = result.get_best_result("loss", mode="min")
        best_config = best_result.config
        del best_result.config["x_train_scale"], 
        del best_result.config["y_train_scale"]
        del best_result.config["x_valid_scale"]
        del best_result.config["y_valid_scale"]
        
        model = create_model(best_result.config)
        
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            model.load_state_dict(torch.load(
                os.path.join(checkpoint_dir, "model.pt"))
                )

        data["loss_epoch"] = best_result.metrics['loss_epoch']
        data["best_train_loss"] = best_result.metrics['loss']
        
    return model, data, best_config


def loss(config):
    # Create the model
    model = create_model(config)
    
    # Train with train dataset
    trainer = Trainer(config, model)
    model = trainer.train(config['x_train_scale'], config['y_train_scale'],
                          config['x_valid_scale'], config['y_valid_scale'])
        

# Convert config to tune config (search space) and add data to tune_config
def config_to_search_space(config, x_train, y_train, x_valid, y_valid):
    
    # By default this config is used for manual hyperparam optimization
    search_space = copy.deepcopy(config)
    search_space["is_manual_optim"] = True
    search_space["optim_hyperparameter"] = []

    
    # Then check if this is for automatic hyper parameter optimization
    for key in config.keys(): 
        
        # Use try because some items are not string or iterable
        try:
            if 'tune.' in config[key]:
                search_space["is_manual_optim"] = False
                search_space["optim_hyperparameter"].append(key)
                search_space[key] = eval(config[key])
        except Exception:
            pass

    # Add train and validation data to search space
    search_space["x_train_scale"] = x_train
    search_space["y_train_scale"] = y_train
    search_space["x_valid_scale"] = x_valid
    search_space["y_valid_scale"] = y_valid
    
    # Retun results of the check and also the keys of hyperparameters    
    return search_space
  
    
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