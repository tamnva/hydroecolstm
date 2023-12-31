
from hydroecolstm.utility.scaler import Scaler, get_scaler_name
from hydroecolstm.data.read_data import read_train_test_data
from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.lstm_linears import Lstm_Linears
from hydroecolstm.model.ea_lstm import Ea_Lstm_Linears
from hydroecolstm.model.train import Train


# Function to train and test the model 
def run_train(config_file):
    
    # Load configuration
    config = read_config(config_file)

    # Read and split data
    data = read_train_test_data(config)
    
    # Scale/transformer name for static, dynamic, and target features
    x_scaler_name, y_scaler_name = get_scaler_name(config)
    
    # Scaler/transformer
    x_scaler, y_scaler = Scaler(), Scaler()
    x_scaler.fit(x=data["x_train"], method=x_scaler_name)
    y_scaler.fit(x=data["y_train"], method=y_scaler_name)
    
    # Scale/transform data
    x_train_scale = x_scaler.transform(x=data["x_train"])
    x_test_scale = x_scaler.transform(x=data["x_test"])
    y_train_scale = y_scaler.transform(x=data["y_train"])
    
    # Create the model
    if config["model_class"] == "LSTM":
        model = Lstm_Linears(config)
    else:
        model = Ea_Lstm_Linears(config)
        
    # Train with train dataset
    trainer = Train(config, model)
    model, y_train_scale_simulated = trainer(x=x_train_scale, y=y_train_scale)
    
    # Simulated result with test dataset
    y_test_simulated_scale = model(x_test_scale)
    
    # Inverse scale/transform back simulated result to real scale
    data["y_train_simulated"] = y_scaler.inverse(y_train_scale_simulated)
    data["y_test_simulated"] = y_scaler.inverse(y_test_simulated_scale)
    
    return model, x_scaler, y_scaler, data, config

