
# Import hydroecolstm function
from hydroecolstm.model_run import run_train
from hydroecolstm.utility.plot import plot
from hydroecolstm.data.read_config import read_config

#-----------------------------------------------------------------------------#
#                                Run the model                                #
#-----------------------------------------------------------------------------#
# Configuration file
config_file = "C:/Users/nguyenta/Documents/GitHub/hydroecolstm/examples/configuration_example/config.yml"
config = read_config(config_file)
    
# Train the model => return model, x_scaler, y_scaler, data
model, x_scaler, y_scaler, data = run_train(config)

# Plot train and valid loss with epoch
data["loss"].drop(['epoch'], axis=1).plot()


# Visualize valid and test data
for object_id in config["object_id"]:
    for target in config["target_features"]:
        p = plot(data, object_id=str(object_id), target_feature=target)
        p.show()
                                   
#-----------------------------------------------------------------------------#
#             Work with GUI, use the two lines below to call the GUI          #
#-----------------------------------------------------------------------------#
#from hydroecolstm.interface.main_gui import show_gui
#show_gui()

#-----------------------------------------------------------------------------#
#             Ungagued basins                                                 #
#-----------------------------------------------------------------------------#
from hydroecolstm.data.read_data import read_forecast_data
from hydroecolstm.utility.evaluation_function import EvaluationFunction
import matplotlib.pyplot as plt

forecast_dataset = read_forecast_data(config)
x_forecast_scale = x_scaler.transform(forecast_dataset["x_forecast"])
y_forecast_scale = model.evaluate(x_forecast_scale)
y_forecast = y_scaler.inverse(y_forecast_scale)
            

# Visualize result: train_test_period = "train" or "test"
for object_id in y_forecast.keys():
    plt.plot(forecast_dataset["time_forecast"][object_id], 
             forecast_dataset["y_forecast"][object_id].detach().numpy(), 
             color = 'blue', label = "Observed", alpha=0.9, linewidth=0.75)
    plt.plot(forecast_dataset["time_forecast"][object_id], 
             y_forecast[object_id].detach().numpy(), 
             color = 'red', label = "Simulated", alpha=0.9, linewidth=0.75)
    plt.legend()
    plt.show()

    
objective = EvaluationFunction(config["eval_function"], config['warmup_length'])
objective(data['y_train'], data['y_train_simulated'])
objective(data['y_valid'], data['y_valid_simulated'])
objective(data['y_test'], data['y_test_simulated'])
objective(forecast_dataset['y_forecast'], y_forecast)

import torch
model.load_state_dict(torch.load(config["best_model"][0]))
torch.save(data, "C:/Users/nguyenta/Documents/GitHub/hydroecolstm/examples/data/data.pt")
torch.save(x_scaler, "C:/Users/nguyenta/Documents/GitHub/hydroecolstm/examples/data/x_scaler.pt")
torch.save(y_scaler, "C:/Users/nguyenta/Documents/GitHub/hydroecolstm/examples/data/y_scaler.pt")
