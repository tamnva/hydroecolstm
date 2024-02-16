# -*- coding: utf-8 -*-

import yaml
import pandas as pd

# Function to read the configurtion file
def read_config(config_file):
    
    # Open and read the configureation file
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    # All required keywords
    keys = ["object_id","input_dynamic_features", "target_features", 
           "train_period", "test_period",  "n_epochs", "learning_rate", 
           "dynamic_data_file"]
 
    # Check if required keywords are missing
    for key in keys: 
        if key not in config.keys():
            raise NameError(f"Error in configuration file, keyword '{key}' is missing")
            
    # Convert date to pandas date time
    if "train_period" in config.keys():
        config["train_period"] = pd.to_datetime(config["train_period"], 
                                                format = "%Y-%m-%d %H:%M")
    if "valid_period" in config.keys():
        config["valid_period"] = pd.to_datetime(config["valid_period"], 
                                            format = "%Y-%m-%d %H:%M")
    if "test_period" in config.keys():
        config["test_period"] = pd.to_datetime(config["test_period"], 
                                           format = "%Y-%m-%d %H:%M")
    if "forecast_period" in config.keys():
        config["forecast_period"] = pd.to_datetime(config["forecast_period"], 
                                               format = "%Y-%m-%d %H:%M")

    # Return output
    return config