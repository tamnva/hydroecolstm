
from hydroecolstm.model_run import run_config
from hydroecolstm.utility.plot import plot
from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.create_model import create_model
from hydroecolstm.interface.utility import write_yml_file
from hydroecolstm.utility.evaluation_function import EvaluationFunction
import matplotlib.pyplot as plt
from pathlib import Path
import torch

#-----------------------------------------------------------------------------#
#                        Set up, train, test model                            #
#-----------------------------------------------------------------------------#

# Read configuration file, please modify the path to the config.yml file
config = read_config("C:/hydroecolstm/examples/2_streamflow_isotope_simulation/config.yml")

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

# Save all data and model state dicts to the output_directory
torch.save(data, Path(config["output_directory"][0], "data.pt"))
torch.save(model.state_dict(), 
           Path(config["output_directory"][0], "model_state_dict.pt"))
write_yml_file(config = best_config, 
               out_file=Path(config["output_directory"][0], "best_config.yml"))

#-----------------------------------------------------------------------------#
#                   Incase you close this file and open again,                #
#                    you can load your data, model as follows                 #
#-----------------------------------------------------------------------------#
config = read_config("C:/hydroecolstm/examples/2_streamflow_isotope_simulation/results/best_config.yml")
model = create_model(config)
model.load_state_dict(torch.load(Path(config["output_directory"][0], 
                                      "model_state_dict.pt")))
data = torch.load(Path(config["output_directory"][0], "data.pt"))