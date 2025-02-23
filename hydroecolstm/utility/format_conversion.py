import pandas as pd
import torch
import numpy as np

#-----------------------------------------------------------------------------#
#              Convert dict of tensor to pandas data frame                    #
#-----------------------------------------------------------------------------#
def tensor_to_pandas_df(tensor:dict[str, torch.Tensor],
                        time:np.array) -> pd.core.frame.DataFrame:
    
    tensor_copy = tensor.copy()
    
    for key in tensor_copy.keys():           
        tensor_copy[key] = tensor_copy[key].numpy().flatten()

    output = pd.DataFrame(tensor_copy)
    output.index = time
    
    return output
    
def nse_df(q_observed, q_simulated):
    
    q_obs = q_observed.copy()
    q_sim = q_simulated.copy()
    
    nse_val = {}
    for column in q_obs.columns:
        nse_val[column] =  [1 - (
            np.nansum((q_obs[column].to_numpy() - 
                    q_sim[column].to_numpy()) ** 2) /
            np.nansum((q_obs[column].to_numpy() - 
                    np.nanmean(q_obs[column].to_numpy())) ** 2)
            )]
        
    output = pd.DataFrame(nse_val).transpose()
    output.columns = ["NSE"]
        
    return output