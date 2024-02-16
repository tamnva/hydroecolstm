
# Import hydroecolstm function
from hydroecolstm.model_run import run_train
from hydroecolstm.utility.plot import plot
from hydroecolstm.data.read_config import read_config
from hydroecolstm.utility.evaluation_function import EvaluationFunction

#-----------------------------------------------------------------------------#
#                                Run the model                                #
#-----------------------------------------------------------------------------#
# Read configuration file
config = read_config("C:/example/1_streamflow_simulation/config.yml")


model, x_scaler, y_scaler, data = run_train(config)

#data.update(test)
data["trainer"].loss.drop(['epoch', 'best_model'], axis=1).plot()
objective = EvaluationFunction("MAE", config['warmup_length'])


print(objective(data['y_train'], data['y_train_simulated']))
print(objective(data['y_valid'], data['y_valid_simulated']))
print(objective(data['y_test'], data['y_test_simulated']))



# Visualize valid and test data
for object_id in config["object_id"]:
    for target in config["target_features"]:
        p = plot(data, object_id=str(object_id), target_feature=target)
        p.show()
                   
#-----------------------------------------------------------------------------#
#             Work with GUI, use the two lines below to call the GUI          #
#-----------------------------------------------------------------------------#
from hydroecolstm.interface.main_gui import show_gui
show_gui()

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

objective(forecast_dataset['y_forecast'], y_forecast)
import torch
torch.save(data, "C:/Users/nguyenta/Documents/GitHub/hydroecolstm/examples/2_streamflow_isotope_simulation/results/data.pt")
test = torch.load("C:/Users/nguyenta/Documents/GitHub/hydroecolstm/examples/2_streamflow_isotope_simulation/results/data.pt")
