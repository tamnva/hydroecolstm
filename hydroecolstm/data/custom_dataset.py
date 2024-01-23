import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, 
                 X:dict[str:torch.Tensor], 
                 Y:dict[str:torch.Tensor], 
                 sequence_length:int, 
                 overlap:int=0):
        
        self.X = None
        self.Y = None
        self.overlap = overlap
        self.sequence_length = sequence_length   
        
        # Combine 2D time series of all basins (keys)
        for key in X.keys():
            Y[key][:self.overlap,:] = float("nan")
            if self.X is None:
                self.X = X[key]
                self.Y = Y[key]
            else:
                self.X = torch.cat((self.X, X[key]), dim = 0)
                self.Y = torch.cat((self.Y, Y[key]), dim = 0)
                    
    def __getitem__(self, index):
        # Get starting and ending index of each mini-batch
        istart = index*(self.sequence_length - self.overlap)
        iend = istart + self.sequence_length
            
        # Mini batch data
        if iend <= len(self.X):
            _x = self.X[istart:iend,:]
            _y = self.Y[istart:iend,:]
        else:
            _x = self.X[-self.sequence_length:,:]
            _y = self.Y[-self.sequence_length:,:]
        
        # Return mini batch consist of input, output
        return _x, _y
    
    def __len__(self):
        # The number of batches (round up)
        num_batches = int(round((len(self.X) - self.overlap)/
                          (self.sequence_length - self.overlap) + 0.49))                          
        return num_batches
