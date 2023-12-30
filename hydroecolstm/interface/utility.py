
def config_to_text(config):
    out_text = []
    for key in config.keys():   
        # Write list object in multiple lines             
        if type(config[key]) is list:
            out_text.append(key + ":\n")
            for element in config[key]:
                out_text.append("  - " + str(element) + "\n")
                
        elif type(config[key]) is dict:
            config_key = config[key]
            out_text.append(key + ":\n")
            
            for key in config_key.keys():
                if type(config_key[key]) is list:
                    out_text.append("  " + key + ":\n")
                    for element in config_key[key]:
                        out_text.append("    - " + str(element) + "\n")
                else:
                    out_text.append("  " + key +": " + str(config_key[key]) + "\n")
        else:
            try:
                # Convert time in config to YYYY-MM-DD HH:MM
                if (config[key].shape[0] == 2):
                    out_text.append(key +": \n")
                    out_text.append("  - " + str(config["train_period"][0])[:16] + "\n")
                    out_text.append("  - " + str(config["train_period"][1])[:16] + "\n")
                    #out_text.append("\n")
            except:
                # Non list object writte in 1 line
                out_text.append(key +": " + str(config[key]) + "\n")
                #out_text.append("\n")
                
    return out_text
    
def sort_key(config):
    config_sort = {}
        
    if "dynamic_data_file" in config.keys():
        config_sort["dynamic_data_file"] = config["dynamic_data_file"]

    if "static_data_file" in config.keys():
        config_sort["static_data_file"] = config["static_data_file"]

    if "input_static_features" in config.keys():
        config_sort["input_static_features"] = config["input_static_features"] 

    if "input_dynamic_features" in config.keys():
        config_sort["input_dynamic_features"] = config["input_dynamic_features"]        

    if "target_features" in config.keys():
        config_sort["target_features"] = config["target_features"] 

    if "object_id" in config.keys():
        config_sort["object_id"] = config["object_id"]        

    if "train_period" in config.keys():
        config_sort["train_period"] = config["train_period"]

    if "test_period" in config.keys():
        config_sort["test_period"] = config["test_period"]

    if "REG" in config.keys():
        config_sort["REG"] = config["REG"]

    if "scaler_input_dynamic_features" in config.keys():
        config_sort["scaler_input_dynamic_features"] = config["scaler_input_dynamic_features"]        

    if "scaler_input_static_features" in config.keys():
        config_sort["scaler_input_static_features"] = config["scaler_input_static_features"] 

    if "scaler_target_features" in config.keys():
        config_sort["scaler_target_features"] = config["scaler_target_features"]        

    if "hidden_size" in config.keys():
        config_sort["hidden_size"] = config["hidden_size"] 

    if "num_layers" in config.keys():
        config_sort["num_layers"] = config["num_layers"]        
        
    if "n_epochs" in config.keys():
        config_sort["n_epochs"] = config["n_epochs"] 

    if "learning_rate" in config.keys():
        config_sort["learning_rate"] = config["learning_rate"]        

    if "dropout" in config.keys():
        config_sort["dropout"] = config["dropout"] 

    if "warmup_length" in config.keys():
        config_sort["warmup_length"] = config["warmup_length"]        

    if "optim_method" in config.keys():
        config_sort["optim_method"] = config["optim_method"] 

    if "objective_function_name" in config.keys():
        config_sort["objective_function_name"] = config["objective_function_name"]        

    if "output_dir" in config.keys():
        config_sort["output_dir"] = config["output_dir"]

    if "output_dir" in config.keys():
        config_sort["output_dir"] = config["output_dir"] 

    if "static_data_file_forecast" in config.keys():
        config_sort["static_data_file_forecast"] = config["static_data_file_forecast"]        

    if "forecast_period" in config.keys():
        config_sort["forecast_period"] = config["forecast_period"] 

    if "object_id_forecast" in config.keys():
        config_sort["object_id_forecast"] = config["object_id_forecast"]
            
    return config_sort

def write_yml_file(config, out_file):
    # Convert config to text
    output_text = config_to_text(config=sort_key(config))
        
    # Write config to config file
    with open(out_file, "w") as config_file:
        for line in output_text:
            config_file.write(line)