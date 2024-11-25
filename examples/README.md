### Data description

In each example folder contains the following subfolders and files

- "./data" folder 
  -  "static_attributes.csv": static catchment attributes
  - "time_series.csv": catchment average time series data at daily time step
- ./results folder
  - data.pt is the dict data types, contains simulated data when running the model with ./config.yml file
  - model.pt is the best model state dicts
- config.yml
  - The configuration file to run this example
- best_config.yml
  - Incase of optimization with ray, the model will try to find the best configuration in the config.yml file. The result (best configuration from optimization with ray was saved as best_config.yml)

- main.py
  - The python script to run this example with the ./config file

### Run the model with and without the GUI

- Run without GUI: Go to the example folder, open the file main.py and change the directory of the config.yml file as well as the link to the data in the config.yml file
- Run with GUI: Open the HydroEcoLSTM GUI and input with settings as shown in the config.yml file in the respective example folder