#!/usr/bin/env python

from hydroecolstm.data.read_data import read_train_valid_test_data
from hydroecolstm.data.scaler import Scaler, get_scaler_name
from hydroecolstm.model.lstm_linears import Lstm_Linears
from hydroecolstm.model.ea_lstm import Ea_Lstm_Linears
from hydroecolstm.train.trainer import Trainer

# Function to train and test the model 
def run_train(config):

    # Read and split data
    data = read_train_valid_test_data(config)
    
    # Scale/transformer name for static, dynamic, and target features
    x_scaler_name, y_scaler_name = get_scaler_name(config)
    
    # Scaler/transformer
    x_scaler, y_scaler = Scaler(), Scaler()
    x_scaler.fit(x=data["x_train"], method=x_scaler_name)
    y_scaler.fit(x=data["y_train"], method=y_scaler_name)
    
    # Scale/transform data input data
    x_train_scale = x_scaler.transform(x=data["x_train"])
    x_valid_scale = x_scaler.transform(x=data["x_valid"])
    x_test_scale = x_scaler.transform(x=data["x_test"])

    # Scale/transform data target train data    
    y_train_scale = y_scaler.transform(x=data["y_train"])
    y_valid_scale = y_scaler.transform(x=data["y_valid"])
    
    # Create the model
    if config["model_class"] == "LSTM":
        model = Lstm_Linears(config)
    else:
        model = Ea_Lstm_Linears(config)
        
    # Train with train dataset
    trainer = Trainer(config, model)
    model = trainer.train(x_train_scale, y_train_scale, x_valid_scale, y_valid_scale)
    
    # Run with input with Dict[str:torch.Tensor] and torch.no_grad()
    y_train_simulated_scale = model.evaluate(x_train_scale)
    y_valid_simulated_scale = model.evaluate(x_valid_scale)
    y_test_simulated_scale = model.evaluate(x_test_scale)
    
    # Inverse scale/transform back simulated result to real scale
    data["y_train_simulated"] = y_scaler.inverse(y_train_simulated_scale)
    data["y_valid_simulated"] = y_scaler.inverse(y_valid_simulated_scale)
    data["y_test_simulated"] = y_scaler.inverse(y_test_simulated_scale)
    data["trainer"] = trainer
    data["y_scaler"] = y_scaler
    data["x_scaler"] = x_scaler
    
    return model, x_scaler, y_scaler, data