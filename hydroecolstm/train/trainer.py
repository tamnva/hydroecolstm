import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from hydroecolstm.train.custom_loss import CustomLoss
from hydroecolstm.data.custom_dataset import CustomDataset

# LSTM + Linears
class Trainer():
    def __init__(self, config, model):

        # Training parameters
        self.lr = config["learning_rate"]
        self.loss_function = CustomLoss(config["loss_function"])
        self.n_epochs = config["n_epochs"]
        self.warmup_length = config["warmup_length"]
        self.sequence_length = config["sequence_length"]
        self.batch_size = config["batch_size"]

        # Model
        self.model = model
        self.patience = config["patience"]
        self.out_dir = config["output_directory"][0]
        
        # Train and loss
        self.loss = None
        
    # Train function
    def train(self, 
              x_train: torch.Tensor, y_train: torch.Tensor,
              x_valid: torch.Tensor, y_valid: torch.Tensor,):
        
        # Optimization function
        optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                
        # Create custom dataset
        xy_train = CustomDataset(x_train, y_train, self.sequence_length,
                                 self.warmup_length)
        xy_valid = CustomDataset(x_valid, y_valid, self.sequence_length,
                                 self.warmup_length)
        
        print("Number of iteration per epoch = ",
              int(xy_train.__len__()/self.batch_size + 0.49))
        
        # Train and valid loss per epoch
        train_loss_epoch = []
        valid_loss_epoch = []
        model_flag = []
        
        # initialize early stoping
        early_stopping = EarlyStopping(patience=self.patience, verbose=False,
                                       path=self.out_dir)
        
        # Train the model
        for epoch in range(self.n_epochs):
            
            # Create batch data for each epoch
            xy_train_batch = DataLoader(xy_train, self.batch_size, shuffle=True)
            xy_valid_batch = DataLoader(xy_valid, self.batch_size, shuffle=True)
            
            # Create list to store train and valid loss per batch
            train_loss_batch = []
            valid_loss_batch = []
            
            # Set model to train mode
            self.model.train()
            
            # Loop over batches
            for x_batch, y_batch in xy_train_batch:
                
                # TODO: why with 3D tensor doesn't work, remove those 2 lines
                # -------------------------------------------------------------
                x_batch = x_batch.view(-1, x_batch.size(2))
                y_batch = y_batch.view(-1, y_batch.size(2))
                # -------------------------------------------------------------
                    
                # Get model output
                y_predict = self.model(x_batch)

                # Reset the gradients to zero
                optim.zero_grad()
                
                # Loss value    
                loss = self.loss_function(y_batch, y_predict)
                 
                # Backward prop
                loss.backward()
                    
                # Update weights and biases
                optim.step()
                
                # Save traning loss 
                train_loss_batch.append(loss.item())
                
            # Set model to eval mode (in this mode, dropout = 0, no normlization)
            self.model.eval()

            # Loop over batches
            for x_batch, y_batch in xy_valid_batch:
                
                # TODO: why with 3D tensor doesn't work, remove those 2 lines
                x_batch = x_batch.view(-1, x_batch.size(2))
                y_batch = y_batch.view(-1, y_batch.size(2))
                
                # Forward pass:
                y_predict = self.model(x_batch)
                
                # Get Loss
                loss = self.loss_function(y_batch, y_predict)
                
                # Save traning loss 
                valid_loss_batch.append(loss.item())

            # Store average loss per epoch for training and validation
            train_loss_epoch.append(np.average(train_loss_batch))
            valid_loss_epoch.append(np.average(valid_loss_batch))
            
            print(f"Epoch [{epoch+1}/{self.n_epochs}], ", 
                  f"average_train_loss = {train_loss_epoch[-1]:.8f}, ",
                  f"avearge_valid_loss = {valid_loss_epoch[-1]:.8f}")
                
            # Early stopping based on validation loss and make checkpoint
            flag = early_stopping(valid_loss_epoch[-1], self.model)
            model_flag.append(flag)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        # If the model does not stops until the last epoch
        # it means that the best model is from last epoch
        if (epoch + 1) == self.n_epochs:
            print("Validation loss continue decreasing. Saving model ...")
            torch.save(self.model.state_dict(), 
                       Path(self.out_dir, "best_model.pt"))
        else:
            # Load the last checkpoint with the best model
            self.model.load_state_dict(torch.load(Path(self.out_dir, 
                                                       "best_model.pt")))
            # Set model to eval mode
            self.model.eval()
            pass
            
        self.loss = pd.DataFrame({"epoch": list(range(1,len(train_loss_epoch)+1)),
                                  'train_loss': train_loss_epoch,
                                  'validation_loss': valid_loss_epoch,
                                  'best_model': model_flag})

        return self.model
    
# ----------------------------------------------------------------------------#
# The EarlyStopping code was copied from                                     #
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
# MIT License                                                                 #
# Copyright (c) 2018 Bjarte Mehus Sunde                                       #
#                                                                             #                                       #
# ----------------------------------------------------------------------------#
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path:str=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = Path(path, "best_model.pt")
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss
        flag = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            flag = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            flag = True
        return flag

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss