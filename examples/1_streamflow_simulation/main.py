
from hydroecolstm.model_run import run_config
from hydroecolstm.utility.plot import plot
from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.create_model import create_model
from hydroecolstm.utility.evaluation_function import EvaluationFunction
from hydroecolstm.data.read_data import read_forecast_data
import matplotlib.pyplot as plt
from pathlib import Path
import torch

#-----------------------------------------------------------------------------#
#                        Set up, train, test model                            #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
config = read_config("C:/hydroecolstm/examples/1_streamflow_simulation/config.yml")

# Create model and train from config 
model, data, best_config = run_config(config)

# Evaluate the model and transform to normal scale
data['y_train_simulated'] = data["y_scaler"].inverse(model.evaluate(data["x_train_scale"]))
data['y_valid_simulated'] = data["y_scaler"].inverse(model.evaluate(data["x_valid_scale"]))
data['y_test_simulated'] = data["y_scaler"].inverse(model.evaluate(data["x_test_scale"]))

# Plot train and validation loss with epoch
data["loss_epoch"].plot()
plt.show() 

# Want to see all keys in data
data.keys()

# Objective function values: MAE, NSE, RMSE, MSE
objective = EvaluationFunction(config['eval_function'], config['warmup_length'])
objective(data['y_train'], data['y_train_simulated'])
objective(data['y_valid'], data['y_valid_simulated'])
objective(data['y_test'], data['y_test_simulated'])

# Visualize valid and test data
for object_id in config["object_id"]:
    for target in config["target_features"]:
        p = plot(data, object_id=str(object_id), target_feature=target)
        p.show()

# Application of the model for (assumed) ungagued basins
forecast_dataset = read_forecast_data(config)
x_forecast_scale = data["x_scaler"].transform(forecast_dataset["x_forecast"])
y_forecast_scale = model.evaluate(x_forecast_scale)

# Application of the model for (assumed) ungagued basins
forecast_dataset = read_forecast_data(config)
x_forecast_scale = data["x_scaler"].transform(forecast_dataset["x_forecast"])
y_forecast_scale = model.evaluate(x_forecast_scale)
y_forecast_simulated = data["y_scaler"].inverse(y_forecast_scale)

# Visualize result: train_test_period = "train" or "test"
for object_id in y_forecast_simulated.keys():
    plt.plot(forecast_dataset["time_forecast"][object_id],
             forecast_dataset["y_forecast"][object_id].detach().numpy(),
             color = 'blue', label = "Observed", alpha=0.9, linewidth=0.75)
    plt.plot(forecast_dataset["time_forecast"][object_id],
             y_forecast_simulated[object_id].detach().numpy(),
             color = 'red', label = "Simulated", alpha=0.9, linewidth=0.75)
    plt.legend()
    plt.show()

# Objective function for forecast
objective(forecast_dataset['y_forecast'], y_forecast_simulated)

# Save all data and model state dicts to the output_directory
torch.save(data, Path(config["output_directory"][0], "data.pt"))
torch.save(model.state_dict(), 
           Path(config["output_directory"][0], "model_state_dict.pt"))

#-----------------------------------------------------------------------------#
#                   Incase you close this file and open again,                #
#                    you can load your data, model as follows                 #
#-----------------------------------------------------------------------------#
config = read_config("C:/hydroecolstm/examples/1_streamflow_simulation/config.yml")

model = create_model(config)
model.load_state_dict(torch.load(Path(config["output_directory"][0], 
                                      "model_state_dict.pt")))

data = torch.load(Path(config["output_directory"][0], "data.pt"))
