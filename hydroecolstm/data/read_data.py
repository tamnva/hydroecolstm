import pandas as pd
import torch

# Read time series data into pandas data frame
def read_split(config:dict=None) -> dict:
    
    # Read input data  
    dynamic_data = pd.read_csv(config['dynamic_data_file'][0], delimiter=",", header=0) 
    dynamic_data["time"] = pd.to_datetime(dynamic_data["time"], format = "%Y-%m-%d")
    
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
    
    config["train_period"] = pd.to_datetime(config["train_period"], format = "%Y-%m-%d")
    train_data = dynamic_data[(dynamic_data["time"] >= config["train_period"][0]) &
                              (dynamic_data["time"] <= config["train_period"][1])]

    # split to train data by object id
    x_train = _split_by_object_id(train_data[config['input_dynamic_features']], config)
    #x_train_column_name = config['input_dynamic_features']
    
    y_train = _split_by_object_id(train_data[config['target_features']], config)
    #y_train_column_name = config['target_features']
    
    config["test_period"] = pd.to_datetime(config["test_period"], format = "%Y-%m-%d")
    test_data = dynamic_data[(dynamic_data["time"] >= config["test_period"][0]) &
                             (dynamic_data["time"] <= config["test_period"][1])]
    
    x_test = _split_by_object_id(test_data[config['input_dynamic_features']], config)
    #x_test_column_name = "please see 'x_train_column_name'"
    
    y_test = _split_by_object_id(test_data[config['target_features']], config)
    #y_train_column_name = "please see 'y_train_column_name'"
    
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
                    raise NameError(f"Error: missing column '{name}' in static data file \n")
            
            # Subset of dynamic_data - only use the required columns and rows
            static_data = static_data[require_columns]
            static_data.set_index("object_id", inplace=True)
            static_data = torch.tensor(static_data.loc[config["object_id"]].values,
                                       dtype=torch.float32)
            
    else:
        static_data = None
        
    # add static data to x_train and y_train
    if static_data is not None:
        for i, object_id in zip(range(len(x_train)), x_train):
            rep_static_data = static_data[i,].repeat(x_train[object_id].shape[0],1)
            x_train[object_id] = torch.cat((x_train[object_id], rep_static_data), 1)

            rep_static_data = static_data[i,].repeat(x_test[object_id].shape[0],1)
            x_test[object_id] = torch.cat((x_test[object_id], rep_static_data), 1)

    return {"x_train":x_train, "y_train": y_train,
            "x_test":x_test, "y_test": y_test}
  
def _split_by_object_id(data, config):
    output = {}
    for object_id in config["object_id"]: 
        output[str(object_id)] = torch.tensor(data.loc[object_id].values, 
                                              dtype=torch.float32)
    return output
    










    