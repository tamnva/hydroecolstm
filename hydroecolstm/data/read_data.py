import pandas as pd
import numpy as np
import torch

# Read time series data into pandas data frame
def read_train_valid_test_data(config:dict=None) -> dict:
    
    # Read input data  
    dynamic_data = pd.read_csv(config['dynamic_data_file'][0], 
                               delimiter=",", header=0) 
    dynamic_data["time"] = pd.to_datetime(dynamic_data["time"], 
                                          format = "%Y-%m-%d %H:%M")
    
    # The column names must contains the following names
    require_columns = ["object_id","time"]
    require_columns.extend(config['input_dynamic_features'])
    require_columns.extend(config['target_features'])    
    
    # Check if data header contains required names        
    for name in require_columns: 
        if name not in dynamic_data.columns:
            raise NameError(f"Error: missing column '{name}' in dynamic data file \n")
            
    # Subset of dynamic_data - only use the required columns, rows
    dynamic_data = dynamic_data[require_columns]
    dynamic_data.set_index("object_id", inplace=True)
    dynamic_data = dynamic_data.loc[config["object_id"]]
    
    train_data = dynamic_data[(dynamic_data["time"] >= config["train_period"][0]) &
                              (dynamic_data["time"] <= config["train_period"][1])]

    # Colum name of the ouput tensor
    x_column_name = config['input_dynamic_features'].copy()
    y_column_name = config['target_features'].copy()
    
    # split to train data by object id    
    x_train = _split_by_object_id(train_data[x_column_name], config["object_id"])
    y_train = _split_by_object_id(train_data[y_column_name], config["object_id"])
    time_train = _time_by_object_id(train_data, config["object_id"])
 
    valid_data = dynamic_data[(dynamic_data["time"] >= config["valid_period"][0]) &
                             (dynamic_data["time"] <= config["valid_period"][1])]
    x_valid = _split_by_object_id(valid_data[x_column_name], config["object_id"])
    y_valid = _split_by_object_id(valid_data[y_column_name], config["object_id"])
    time_valid = _time_by_object_id(valid_data, config["object_id"])
    

    test_data = dynamic_data[(dynamic_data["time"] >= config["test_period"][0]) &
                             (dynamic_data["time"] <= config["test_period"][1])]
    x_test = _split_by_object_id(test_data[x_column_name], config["object_id"])
    y_test = _split_by_object_id(test_data[y_column_name], config["object_id"])
    time_test = _time_by_object_id(test_data, config["object_id"])
    
    # Read static input data file    
    if 'input_static_features' in config:
        if len(config['input_static_features']) > 0:
            static_data = pd.read_csv(config['static_data_file'][0], delimiter=",", 
                                      header=0)
            # The column names must contains the following names
            require_columns = ["object_id"]
            require_columns.extend(config['input_static_features'])    
            
            # Check if data header contains required names
            for name in require_columns: 
                if name not in static_data.columns:
                    raise NameError(f"Error: missing column '{name}' in static data\n")
            
            # Subset of dynamic_data - only use the required columns and rows
            static_data = static_data[require_columns]
            static_data.set_index("object_id", inplace=True)
            static_data = torch.tensor(static_data.loc[config["object_id"]].values,
                                       dtype=torch.float32)
            
            # Update columne name
            x_column_name.extend(config['input_static_features'])
            
    else:
        static_data = None
        
    # add static data to x_train and y_train
    if static_data is not None:
        for i, object_id in zip(range(len(x_train)), x_train):
            rep_static_data = static_data[i,].repeat(x_train[object_id].shape[0],1)
            x_train[object_id] = torch.cat((x_train[object_id], rep_static_data), 1)

            rep_static_data = static_data[i,].repeat(x_valid[object_id].shape[0],1)
            x_valid[object_id] = torch.cat((x_valid[object_id], rep_static_data), 1)
            
            rep_static_data = static_data[i,].repeat(x_test[object_id].shape[0],1)
            x_test[object_id] = torch.cat((x_test[object_id], rep_static_data), 1)

    return {"x_train":x_train, "y_train": y_train, "time_train" : time_train,
            "x_valid":x_valid, "y_valid": y_valid, "time_valid" : time_valid, 
            "x_test":x_test, "y_test": y_test, "time_test": time_test,
            "x_column_name": x_column_name, "y_column_name": y_column_name}

# -----------------------------------------------------------------------------
def read_forecast_data(config:dict=None) -> dict:
    
    # Read input data & check if users use the same file for train_test and forecast
    if config['dynamic_data_file_forecast'][0] == "dynamic_data_file": 
        dynamic_data = pd.read_csv(config['dynamic_data_file'][0], 
                                   delimiter=",", header=0)
    else:
        dynamic_data = pd.read_csv(config['dynamic_data_file_forecast'][0], 
                                   delimiter=",", header=0)
    
    # Convert date time to pandas date time
    dynamic_data["time"] = pd.to_datetime(dynamic_data["time"], 
                                          format = "%Y-%m-%d %H:%M")
   
    # The column names must contains the following names
    require_columns = ["object_id","time"]
    require_columns.extend(config['input_dynamic_features'])
    
    # Check if data header contains required names        
    for name in require_columns: 
        if name not in dynamic_data.columns:
            raise NameError(f"Error: missing column '{name}' in forecast data\n")

    # Add NaN to the dynamic data target features (if they are in the data) 
    for target_feature in config["target_features"]:
        if target_feature not in dynamic_data.columns:
            dynamic_data[target_feature] = np.nan
    require_columns.extend(config["target_features"])
    
    # Subset of dynamic_data - only selected columns
    dynamic_data = dynamic_data[require_columns]

    # Subset of dynamic data - only selected object_id
    dynamic_data.set_index("object_id", inplace=True)
    dynamic_data = dynamic_data.loc[config["object_id_forecast"]]
    
    # Subset of dynamic data - only forecast period
    forecast_data = dynamic_data[(dynamic_data["time"] >= config["forecast_period"][0]) &
                              (dynamic_data["time"] <= config["forecast_period"][1])]

    # Column name of the ouput tensor
    x_column_name = config['input_dynamic_features'].copy()
    y_column_name = config['target_features'].copy()
    
    # Split to x_forecast and y_forecast by object id
    x_forecast = _split_by_object_id(forecast_data[x_column_name], 
                                     config["object_id_forecast"])
    y_forecast = _split_by_object_id(forecast_data[y_column_name], 
                                     config["object_id_forecast"])
    time_forecast = _time_by_object_id(forecast_data, config["object_id_forecast"])
      
    # Read static input data file    
    if 'input_static_features' in config:
        if len(config['input_static_features']) > 0:
            if config['static_data_file_forecast'][0] == "static_data_file":
                static_data = pd.read_csv(config['static_data_file'][0], 
                                          delimiter=",", header=0)
            else:
                static_data = pd.read_csv(config['static_data_file_forecast'][0], 
                                          delimiter=",", header=0)
                
            # The column names must contains the following names
            require_columns = ["object_id"]
            require_columns.extend(config['input_static_features'])    
            
            # Check if data header contains required names
            for name in require_columns: 
                if name not in static_data.columns:
                    raise NameError(f"Error: missing column '{name}' in" + 
                                    " static data file \n")
            
            # Subset of dynamic_data - only use the required columns and rows
            static_data = static_data[require_columns]
            static_data.set_index("object_id", inplace=True)
            static_data = torch.tensor(static_data.loc[config["object_id_forecast"]].values,
                                       dtype=torch.float32)
            
            # Update columne name
            x_column_name.extend(config['input_static_features'])
            
    else:
        static_data = None

    # add static data to x_forecast and y_forecast
    if static_data is not None:
        for i, object_id in zip(range(len(x_forecast)), x_forecast):
            rep_static_data = static_data[i,].repeat(x_forecast[object_id].shape[0],1)
            x_forecast[object_id] = torch.cat((x_forecast[object_id], rep_static_data), 1)

    return {"x_forecast":x_forecast, "y_forecast": y_forecast, 
            "time_forecast": time_forecast, "x_column_name": x_column_name, 
            "y_column_name": y_column_name}

# split by object id
def _split_by_object_id(data, object_id):
    output = {}
    for objectid in object_id: 
        output[str(objectid)] = torch.tensor(data.loc[objectid].values,
                                             dtype=torch.float32)
    return output

# Get date time by object id
def _time_by_object_id(data, object_id):
    output = {}
    for objectid in object_id: 
        output[str(objectid)] = data.loc[objectid]["time"].values
    return output
    