### 1. What is HydroEcoLSTM

- HydroEcoLSTM is a tool for modelling Hydro-ecological processes with Long short-term Memory (LSTM) neural network. HydroEcoLSTM is provided to users as a python packages with graphical user interface (GUI) and without GUI. 
- This is the beta version, package documentation will be provided in later versions

### 2. How to start?

```python
# Install the package from github using pip command
pip install git+https://github.com/tamnva/hydroecolstm.git

# Import the package and show the GUI
import hydroecolstm
hydroecolstm.interface.show_gui()

# Example of static and dynamic data in this folder
# static data (catchment attributes) 'examples/static_attributes.csv'
# dynamic data (time series input and target features) 'examples/time_series.csv'
```

### 3. The GUI

- After lanching the GUI, you should see the following window (the latest version in the "development" branch could look different)

<p align="center">
  <img src="examples/GUI.gif" width=100% title="hover text">
</p>
