
# Import hydroecolstm function
from hydroecolstm.model_run import run_train
from hydroecolstm.utility.plot import plot

#-----------------------------------------------------------------------------#
#                                Run the model                                #
#-----------------------------------------------------------------------------#
# Configuration file
config_file = "C:/Users/nguyenta/Documents/GitHub/config.yml"

# Train the model => return model, x_scaler, y_scaler, data
model, x_scaler, y_scaler, data, config = run_train(config_file)

# Visualize result: train_test_period = "train" or "test"
for object_id in config["object_id"]:
    for target in config["target_features"]:
        p = plot(data, object_id=str(object_id), train_test_period="test", 
                   target_feature=target)
        p.show()
                                    
#-----------------------------------------------------------------------------#
#             Work with GUI, use the two lines below to call the GUI          #
#-----------------------------------------------------------------------------#
from hydroecolstm.interface.main_gui import show_gui
show_gui()




'''
from hydroecolstm.utility.loss_function import LossFunction
from hydroecolstm.data.read_data import read_forecast_data
import matplotlib.pyplot as plt
# Apply trained model for ungagued basins
forecast_dataset = read_forecast_data(config)
x_forecast_scale = x_scaler.transform(forecast_dataset["x_forecast"])
y_forecast_scale = model(x_forecast_scale)
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
    
objective = LossFunction()
objective(data['y_train'], data['y_train_simulated'], config['warmup_length'], config['objective_function_name'])
objective(data['y_test'], data['y_test_simulated'], config['warmup_length'], config['objective_function_name'])
objective(forecast_dataset['y_forecast'], y_forecast, config['warmup_length'], config['objective_function_name'])

'''
