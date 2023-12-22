
from hydroecolstm.utility.scaler import Scaler, get_scaler_name
from hydroecolstm.data.read_data import read_train_test_data, read_forecast_data
from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.lstms import LSTM_DL
import matplotlib.pyplot as plt


# Read configuration file
config = read_config("C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM/examples/config.yml")


# Read and split data
data = read_train_test_data(config)

# Scale data
x_scaler, y_scaler = get_scaler_name(config)

# Scale x_train
x_train_scaler, y_train_scaler = Scaler(), Scaler()
x_train_scaler.fit(x=data["x_train"], method=x_scaler)
x_train_scale = x_train_scaler.transform(x=data["x_train"])
x_test_scale = x_train_scaler.transform(x=data["x_test"])

y_train_scaler.fit(x=data["y_train"], method=y_scaler)
y_train_scale = y_train_scaler.transform(x=data["y_train"])
y_test_scale = y_train_scaler.transform(x=data["y_test"])

# Model
my_model = LSTM_DL(config=config)
model, y_predict = my_model.train(x_train=x_train_scale, y_train=y_train_scale)
y_test_scale_sim=my_model.forward(x_test_scale)


# Plot
for object_id in y_test_scale.keys():
    obs = y_test_scale[object_id].detach().numpy()
    sim = y_test_scale_sim[object_id].detach().numpy()
    plt.plot(sim[:,0], color = 'blue', label = "Simulated Q (train)", alpha=0.9, linewidth=0.75)
    plt.plot(obs[:,0], color = 'red', label = "Simulated Q (test)", alpha=0.9, linewidth=0.75)
    plt.title(label=f"object_id = {object_id}, target featue = {config['target_features'][0]}")
    plt.show()


# Work with GUI, use the two lines below to call the GUI
from hydroecolstm.interface.main_gui import show_gui
show_gui()

'''
config = read_config("C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM/examples/config.yml")
test = config["test_basins"]
test[0]['dynamic_data_file_test']
'''