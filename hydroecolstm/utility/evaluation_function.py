import torch

class EvaluationFunction():
    def __init__(self, function_name:str, nskip:int):
        
        # Dict of all available evaluation functions
        evaluation_functions = {"MSE": self.MSE, "RMSE": self.RMSE,
                                "NSE": self.NSE, "MAE": self.MAE}
        
        # Selected evaluation function
        self.eval_function = evaluation_functions[function_name]
        self.nskip = nskip
        
    def __call__(self, y_true:torch.Tensor, y_predict:torch.Tensor) -> torch.Tensor:
        
        # Get evaluation values for each basins (key), each target variables
        eval_values = {}

        for key in y_true.keys():
            eval_values[key] = self.eval_function(y_true[key][self.nskip:,],
                                                  y_predict[key][self.nskip:,])

        avg_eval_values = sum(sum(eval_values.values()))/((len(eval_values))*eval_values[next(iter(eval_values))].shape[0])
            
        return eval_values, avg_eval_values
    
    def MSE(self, ytrue:torch.Tensor, ypredict:torch.Tensor):
        mask = ~torch.isnan(ytrue)
        mse = []
        for i in range(ytrue.shape[1]):
            mse.append(torch.mean((ytrue[:,i][mask[:,i]] - ypredict[:,i][mask[:,i]])**2))
        mse = torch.stack(mse)
        return mse


    def RMSE(self, ytrue:torch.Tensor, ypredict:torch.Tensor):
        mse = self.MSE(ytrue, ypredict)
        rmse = mse**0.5
        return rmse
    
    # 1 - Nash–Sutcliffe efficiency (NSE)
    def NSE(self, ytrue:torch.Tensor, ypredict:torch.Tensor):
        mask = ~torch.isnan(ytrue)
        
        # Sum of Square Error (sse) = sum((true-predict)^2)
        # Sum of Square Difference around mean (ssd) = sum((true-mean_true)^2)
        sse = []        
        ssd = []
        for i in range(ytrue.shape[1]):
            sse.append(torch.sum((ytrue[:,i][mask[:,i]] - ypredict[:,i][mask[:,i]])**2))
            ssd.append(torch.sum((ytrue[:,i][mask[:,i]] - torch.nanmean(ytrue[:,i]))**2))
        
        # get 1 - nse, here I call it as nse
        nse = 1.0 - torch.stack(sse)/torch.stack(ssd)
        
        if torch.isnan(nse).any():
            raise ValueError("nan values found when calculating NSE - zero division")
            
        return nse
       
    def MAE(self, ytrue:torch.Tensor, ypredict:torch.Tensor):
        mask = ~torch.isnan(ytrue)
        mae = []
        for i in range(ytrue.shape[1]):
            error = ytrue[:,i][mask[:,i]] - ypredict[:,i][mask[:,i]]
            mae.append(torch.mean(torch.abs(error)))
        mae = torch.stack(mae)
        
        return mae
