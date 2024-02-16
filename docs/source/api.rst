Examples working without GUI
============================

Streamflow simulation
---------------------

.. code-block:: python

   # Import hydroecolstm function
   from hydroecolstm.model_run import run_train
   from hydroecolstm.utility.plot import plot
   from hydroecolstm.data.read_config import read_config
   from hydroecolstm.utility.evaluation_function import EvaluationFunction

   # Read configuration file:
   config = read_config("C:/example/1_streamflow_simulation/config.yml")
   
   # Create and train the model
   model, x_scaler, y_scaler, data = run_train(config)

   # Plot train and validation loss
   data["trainer"].loss.drop(['epoch', 'best_model'], axis=1).plot()
   
   # Get objective function value, e.g., Mean Square Error (MSE)
   objective = EvaluationFunction("MAE", config['warmup_length'])
   
   objective(data['y_train'], data['y_train_simulated']
   objective(data['y_valid'], data['y_valid_simulated'])
   objective(data['y_test'], data['y_test_simulated'])



   # Visualize valid and test data
   for object_id in config["object_id"]:
       for target in config["target_features"]:
           p = plot(data, object_id=str(object_id), target_feature=target)
           p.show()