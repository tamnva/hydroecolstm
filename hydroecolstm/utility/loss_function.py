import torch

class LossFunction:
    def __call__(self, y_true:torch.Tensor, y_predict:torch.Tensor, nskip: int,
                 objective_function_name:str) -> torch.Tensor:
        
        # Loss function as list
        loss_functions = {"MSE": self.MSE, "RMSE": self.RMSE, 
                          "1-NSE": self.NSE, "MAE": self.MAE}
        
        loss = {}

        for key in y_true.keys():
            loss[key] = loss_functions[objective_function_name](y_true[key][nskip:,],
                                                                y_predict[key][nskip:,])

        avg_loss = sum(sum(loss.values()))/((len(loss))*loss[next(iter(loss))].shape[0])
        
        # Raise value error nan loss
        # if torch.isnan(avg_loss):
        #    raise ValueError("nan values found when calculating loss value")
            
        return loss, avg_loss
    
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
        nse = torch.stack(sse)/torch.stack(ssd)
        
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
        
        
        