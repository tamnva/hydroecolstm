# -*- coding: utf-8 -*-

import yaml

# Function to read the configurtion file
def read_config(config_file):
    
    # Open and read the configureation file
    with open(config_file, 'r') as file:
        cfg = yaml.safe_load(file)
    
    # All required keywords
    key = ["object_id","input_dynamic_features", "target_features", 
           "train_period", "test_period",  "n_epochs", "learning_rate", 
           "dynamic_data_file"]
 
    # Check if required keywords are missing
    for keyword in key: 
        if keyword not in cfg:
            raise NameError(f"Error in configuration file, keyword '{keyword}' is missing")
            
    #checkiftraiistest

    # Return output
    return cfg