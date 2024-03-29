Examples
========

Streamflow simulation
---------------------
  
The configuration file ``config.yml`` and data for this example can be found `here <https://github.com/tamnva/hydroecolstm/tree/master/examples/1_streamflow_simulation>`_. 

.. code-block:: python

   # Import hydroecolstm function
   from hydroecolstm.model_run import run_train
   from hydroecolstm.utility.plot import plot
   from hydroecolstm.data.read_config import read_config
   from hydroecolstm.utility.evaluation_function import EvaluationFunction
   from hydroecolstm.data.read_data import read_forecast_data
   import matplotlib.pyplot as plt
   from pathlib import Path
   import torch

   # Read configuration file
   # Please modify the path to the config.yml and link to data
   config = read_config("C:/example/1_streamflow_simulation/config.yml")

   # Create model and train
   model, x_scaler, y_scaler, data = run_train(config)

   # Plot training and validation losses
   data["trainer"].loss.drop(['epoch', 'best_model'], axis=1).plot()

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
   x_forecast_scale = x_scaler.transform(forecast_dataset["x_forecast"])
   y_forecast_scale = model.evaluate(x_forecast_scale)
   
   # Application of the model for (assumed) ungagued basins
   forecast_dataset = read_forecast_data(config)
   x_forecast_scale = x_scaler.transform(forecast_dataset["x_forecast"])
   y_forecast_scale = model.evaluate(x_forecast_scale)
   y_forecast_simulated = y_scaler.inverse(y_forecast_scale)

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
   #torch.save(model.state_dict(), Path(config["output_directory"][0], "model.pt"))


Multiple outputs simulation
---------------------------
  
The configuration file ``config.yml`` and data for this example can be found `here <https://github.com/tamnva/hydroecolstm/tree/master/examples/2_streamflow_isotope_simulation>`_. 

.. code-block:: python

   # Import hydroecolstm function
   from hydroecolstm.model_run import run_train
   from hydroecolstm.utility.plot import plot
   from hydroecolstm.data.read_config import read_config
   from hydroecolstm.utility.evaluation_function import EvaluationFunction
   from hydroecolstm.data.read_data import read_forecast_data
   import matplotlib.pyplot as plt
   from pathlib import Path
   import torch

   # Read configuration file
   # Please modify the path to the config.yml and link to data
   config = read_config("C:/example/2_streamflow_isotope_simulation/config.yml")

   # Create model and train
   model, x_scaler, y_scaler, data = run_train(config)

   # Plot training and validation losses
   data["trainer"].loss.drop(['epoch', 'best_model'], axis=1).plot()

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
   #torch.save(model.state_dict(), Path(config["output_directory"][0], "model.pt"))
