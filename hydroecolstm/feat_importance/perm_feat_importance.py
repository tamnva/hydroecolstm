
import copy
import torch
from hydroecolstm.utility.evaluation_function import EvaluationFunction

# Permutation feature important basin wise
def pfib(features: str, x_test_scale:dict[str, torch.Tensor], 
         y_test:dict[str, torch.Tensor], y_scaler, 
         trained_model, objective_function_name:str, nskip:int):
    
    # Evaluation function
    objective = EvaluationFunction(function_name=objective_function_name, 
                                   nskip=nskip)
    
    #obj = objective(y_test, 
    #                y_scaler.inverse(trained_model.evaluate(x_test_scale)))
    
    # Loop over features
    for i in range(len(features)): 
        x_perm = {}
        
        for key, x in zip(x_test_scale.keys(), x_test_scale.values()):
            
            # Shuffle index of feature i
            idx = torch.randperm(x.shape[0])
            
            # Shuffle data 
            x_copy = copy.deepcopy(x)
            x_copy[:,i] = x_copy[idx, i]
            
            # Save permutated data for each key
            x_perm[key] = copy.deepcopy(x_copy)
            
        prediction = y_scaler.inverse(trained_model.evaluate(x_perm))
        
        if i == 0: 
            output = objective(y_test, prediction)
            output.rename(columns={"objective_function_value": features[i]}, 
                      inplace=True)
        else: 
            output[features[i]] =  objective(
                y_test, prediction)["objective_function_value"]
            
    return output  #.subtract(obj['objective_function_value'], axis=0)