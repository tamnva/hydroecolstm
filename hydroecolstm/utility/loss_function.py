import torch

class LossFunction:
    def __call__(self, y_true:torch.Tensor, y_predict:torch.Tensor, nskip: int,
                 objective_function_name:str) -> torch.Tensor:
        
        # Loss function as list
        loss_functions = {"MSE": self.MSE, "RMSE": self.RMSE}
        
        loss = {}

        for key in y_true.keys():
            loss[key] = loss_functions[objective_function_name](y_true[key][nskip:,],
                                                                y_predict[key][nskip:,])

        avg_loss = sum(sum(loss.values()))/((len(loss))*loss[next(iter(loss))].shape[0])
        
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