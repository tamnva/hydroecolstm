import torch
from torch import nn

class CustomLoss(nn.Module):
    def __init__(self, loss_function:str):
        super(CustomLoss, self).__init__()
        
        # Dict of all available loss functions
        loss_functions = {"MSE": self.MSE, 
                          "RMSE": self.RMSE,
                          "MAE": self.MAE}
        
        # Use this loss function

        self.loss_function = loss_functions[loss_function]
    
    def forward(self, y_true:torch.Tensor, y_predict:torch.Tensor) -> torch.Tensor:
        
        # TODO: Why loss with 3D differnt with 2D Tensor
        # y_true = y_true.view(-1, y_true.size(2))
        # y_predict = y_predict.view(-1, y_predict.size(2))
                
        mask = ~torch.isnan(y_true)
        loss = self.loss_function(y_true, y_predict, mask)
            
        return loss
    
    # Mean square error
    def MSE(self, y_true:torch.Tensor, y_predict:torch.Tensor,
            mask:torch.Tensor)-> torch.Tensor: 

        # Mean square error
        loss_fn = nn.MSELoss()
        mse = loss_fn(y_true[mask], y_predict[mask])
        
        # Return output
        return mse
    
    # Mean absolute error
    def MAE(self, y_true:torch.Tensor, y_predict:torch.Tensor,
            mask:torch.Tensor)-> torch.Tensor: 
        
        # Mean absolute error
        loss_fn = nn.L1Loss()
        mae = loss_fn(y_true[mask], y_predict[mask])
        return mae      
    
    # Root mean square error
    def RMSE(self, y_true:torch.Tensor, y_predict:torch.Tensor,
            mask:torch.Tensor)-> torch.Tensor: 
        
        # Root Mean Square Error
        rmse = self.MSE(y_true, y_predict, mask)**0.5
        return rmse
