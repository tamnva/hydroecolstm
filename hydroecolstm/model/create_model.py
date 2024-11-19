
from hydroecolstm.model.lstm_linears import Lstm_Linears
from hydroecolstm.model.ea_lstm import Ea_Lstm_Linears

def create_model(config, state_dict=None):
    
    # Create the model
    if config["model_class"] == "LSTM":
        model = Lstm_Linears(config)
    else:
        model = Ea_Lstm_Linears(config)   
        
    # Assign state dict
    if state_dict is not None: 
        model.load_state_dict(state_dict)
    
    return model