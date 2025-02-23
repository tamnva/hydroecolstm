
import copy
import torch
import pandas as pd
from hydroecolstm.utility.evaluation_function import EvaluationFunction

#features=features
#x_test_scale=data["x_test_scale"]
#y_test=data["y_test"]
#y_scaler=data["y_scaler"]
#trained_model=model
#objective_function_name="NSE"
#y_column_name=data["y_column_name"]
#nskip=config["warmup_length"]
#seed=100
            
# Permutation feature important basin wise
def pfib(features: str, x_test_scale:dict[str, torch.Tensor], 
         y_test:dict[str, torch.Tensor], y_scaler, 
         trained_model, objective_function_name:str, 
         nskip:int, y_column_name:str, seed:int=None):
    
    # Evaluation function
    objective = EvaluationFunction(function_name=objective_function_name, 
                                   nskip=nskip, y_column_name=y_column_name)
    
    #obj = objective(y_test, 
    #                y_scaler.inverse(trained_model.evaluate(x_test_scale)))
    
    # Loop over features
    for i in range(len(features)): 
        x_perm = {}
        
        for key, x in zip(x_test_scale.keys(), x_test_scale.values()):

            # Shuffle index of feature i
            if seed is not None: 
                torch.manual_seed(0)
                
            idx = torch.randperm(x.shape[0])
            
            # Shuffle data 
            x_copy = copy.deepcopy(x)
            x_copy[:,i] = x_copy[idx, i]
            
            # Save permutated data for each key
            x_perm[key] = copy.deepcopy(x_copy)
            
        prediction = y_scaler.inverse(trained_model.evaluate(x_perm))
        
        if i == 0: 
            output = objective(y_test, prediction)
            output.columns = features[i] + "_" + output.columns 
            #output.rename(columns={output.columns[0]: features[i]}, 
            #          inplace=True)
        else: 
            temp = objective(y_test, prediction)
            temp.columns = features[i] + "_" + temp.columns 
            
            output = pd.concat([output, temp], axis=1)
            #output[features[i]] =  objective(
            #    y_test, prediction)["objective_function_value"]
        
        #output.columns = ["s"]
        
    return output  #.subtract(obj['objective_function_value'], axis=0)


