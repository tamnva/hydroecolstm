
# Import hydroecolstm function
from hydroecolstm.model_run import run_train
from hydroecolstm.utility.plot import plot


#-----------------------------------------------------------------------------#
#                                Run the model                                #
#-----------------------------------------------------------------------------#
# Configuration file
config_file = "C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM/examples/config.yml"

# Train the model => return model, x_scaler, y_scaler, data
model, x_scaler, y_scaler, data, config = run_train(config_file)

# Visualize result: train_test_period = "train" or "test"
for object_id in config["object_id"]:
    plt = plot(data, object_id=str(object_id), train_test_period="test", 
               target_feature="discharge_vol_m3_s")
    plt.show()

                                    
#-----------------------------------------------------------------------------#
#             Work with GUI, use the two lines below to call the GUI          #
#-----------------------------------------------------------------------------#
from hydroecolstm.interface.main_gui import show_gui
show_gui()
