


from hydroecolstm.utility.scaler import Scaler, get_scaler_name
from hydroecolstm.data.read_data import read_split
from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.lstms import LSTM_DL



# Read configuration file
config = read_config("C:/Users/nguyenta/Documents/GitHub/HydroEcoLSTM/examples/config.yml")


# Read and split data
data = read_split(config)

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

x_test_scale
import matplotlib.pyplot as plt


obs = y_test_scale['2011'].detach().numpy()
sim = y_test_scale_sim['2011'].detach().numpy()

#plt.scatter(obs[:,1], sim[:,1])

plt.plot(sim, color = 'blue', label = "Simulated Q (train)", alpha=0.9, linewidth=0.75)
plt.plot(obs, color = 'red', label = "Simulated Q (test)", alpha=0.9, linewidth=0.75)
plt.show()
    

from hydroecolstm.interface.main_gui import show_gui
show_gui()


def get_obs_predict_object_id(object_id):
    obs = y_test_scale['2011'].detach().numpy()
    sim = y_test_scale_sim['2011'].detach().numpy()
    return obs, sim


import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

x = 4 + np.random.normal(0, 2, 24)
y = 4 + np.random.normal(0, 2, len(x))


# create a figure
figure = Figure(figsize=(6, 4), dpi=100)
figure_canvas = FigureCanvasTkAgg(figure, self.right_frame )
NavigationToolbar2Tk(figure_canvas, self.right_frame )

axes = figure.add_subplot()

axes.plot(x, color = 'blue', label = "Predicted (test data)", alpha=0.9, linewidth=0.75)
axes.plot(y, 'ro', label = "Observed (test data)", alpha=0.9, markersize=2.5 )
axes.legend()

figure

figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
