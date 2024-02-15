
import customtkinter as ctk
import matplotlib
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
                                               NavigationToolbar2Tk)


# Plot loss
def plot_train_valid_loss(loss, plot_window, xlabel, ylabel, title, 
                          train_color, train_line_style, train_legend,
                          valid_color, valid_line_style, valid_legend,
                          best_model_legend):

    figure = Figure(figsize=(15, 4.5), dpi=100)
    figure_canvas = FigureCanvasTkAgg(figure, plot_window)
    toolbar = NavigationToolbar2Tk(figure_canvas, plot_window, 
                                   pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
            
    axes = figure.add_subplot()
    epoch = loss.query('best_model == True').iloc[-1,:].epoch
    
    # Default plot setting

    if train_color == "": train_color = "coral"
    if train_line_style == "": train_line_style = "dashed"
    if train_legend == "": train_legend = "Training loss"
    if valid_color == "": valid_color = "blue"
    if valid_line_style == "": valid_line_style = "solid"
    if valid_legend == "": valid_legend = "Validation loss"
    if xlabel == "": xlabel = "Epoch"
    if ylabel == "": ylabel = "Loss"
    if title == "": title = 'Training and validation loss'
    if best_model_legend == "": best_model_legend = "Best model at epoch " + str(epoch)
    
    try:
        axes.plot(loss["epoch"], 
                  loss["train_loss"], 
                  label = train_legend, 
                  linestyle = train_line_style,
                  color = train_color)
    except:
        axes.plot(loss["epoch"], 
                  loss["train_loss"], 
                  label = train_legend, 
                  linestyle = "dashed",
                  color = "coral")
    try:    
        axes.plot(loss["epoch"], 
                  loss["validation_loss"], 
                  label=valid_legend, 
                  linestyle = valid_line_style,
                  color = valid_color)
    except:
        axes.plot(loss["epoch"], 
                  loss["validation_loss"], 
                  label=valid_legend, 
                  color = "blue")
    axes.axvline(x=epoch, 
                 label=best_model_legend,
                 color = "grey",
                 linestyle='dashed')
    
    # Set x label
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend()
            
    figure_canvas.get_tk_widget().pack(side=tk.TOP, 
                                       fill=tk.BOTH, 
                                       expand=1)

# Check color
def check_color(entry, default):
    color  = entry.get().strip()
    if len(color) > 0:
        if color not in matplotlib.colors.cnames.keys():
            tk.messagebox.showinfo(title="Error", message= color +
                                   ": unknown color name, default color is used")
            color = default 
    else:
        color = default
    return color

# Check color
def check_marker(entry, default):
    marker  = entry.get().strip()
    if len(marker) > 0:
        if marker not in [".","o","s","^","v","+","x"]:
            tk.messagebox.showinfo(title="Error", message= marker +
                                   ": unknown marker name, default marker is used")
            marker = default 
    else:
        marker = default
    return marker

# Check color
def check_line_plot(entry):
    line_plot  = entry.get().strip()
    if len(line_plot) > 0:
        if line_plot in ["False","false","FALSE"]:
            line_plot = False
        else:
            line_plot = True
    else:
        line_plot = True
    return line_plot
    
# Check alpha
def check_alpha(entry, default):
    alpha =  entry.get().strip()
    
    if len(alpha) > 0:
        try:
            alpha = float(alpha)
            if alpha > 1.0: alpha = 1.0
            if alpha < 0.0: alpha = 0.0
        except:
            alpha = default
    else:
        alpha = default
    return alpha

# Check line style    
def check_linestyle(entry, default):
    linestyle =  entry.get().strip()
    
    if len(linestyle) > 0:
        if linestyle not in ["solid", "dashed", "dashdot", "dotted"]:
            linestyle = default
    else:
        linestyle = default
        
    return linestyle

# Check line style    
def check_size(entry, default):
    size =  entry.get().strip()
    
    if len(size) > 0:
        try:
            size = float(size)
            if size < 0.01: size = default
        except:
            size = default
    else:
        size = default
    return size

# Check line style    
def check_ylim(entry, default):
    ylim =  entry.get().strip()
    
    if len(ylim) > 0:
        try:
            ylim = float(ylim)
        except:
            ylim = False
    else:
        ylim = False
    return ylim



#-----------------------------------------------------------------------------#
#                                 plot timeseries                             #
#-----------------------------------------------------------------------------#
def plot_time_series(plot_window, data, key, idx, lineplot, 
                 color_obs, alpha_obs, size_obs, linestyle_obs,
                 marker_obs, label_obs, color_sim, alpha_sim, 
                 size_sim, linestyle_sim, label_sim, title, xlabel,
                 ylabel, ylim_up):
    
    # Create figure
    figure = Figure(figsize=(15, 4.5), dpi=100)
    figure_canvas = FigureCanvasTkAgg(figure, plot_window)
    toolbar = NavigationToolbar2Tk(figure_canvas, plot_window, 
                                   pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    axes = figure.add_subplot()
    
    # Key names of time and y variables in the input data
    time_name = ["time_train", "time_valid", "time_test"]
    y_name = ["y_train", "y_valid", "y_test"]
    
    # Plot observed -----------------------------------------------------------
    for i in range(3):
        time = data[time_name[i]][key]
        y_obs = data[y_name[i]][key][:,idx].detach().numpy()
        
        if lineplot:
            if i < 2:
                axes.plot(time, 
                          y_obs, 
                          linestyle=linestyle_obs,
                          color=color_obs, 
                          alpha=alpha_obs,
                          linewidth=size_obs)
            else:
                axes.plot(time, 
                          y_obs, 
                          linestyle=linestyle_obs,
                          color=color_obs, 
                          alpha=alpha_obs, 
                          linewidth=size_obs, 
                          label=label_obs)                
        else:
            if i < 2:
                axes.plot(time, 
                          y_obs, 
                          marker_obs,
                          color=color_obs, 
                          alpha=alpha_obs,
                          markersize=size_obs)
            else:
                axes.plot(time, 
                          y_obs, 
                          marker_obs,
                          color=color_obs, 
                          alpha=alpha_obs,
                          markersize=size_obs,
                          label=label_obs)
            
            
    # Plot simulated ----------------------------------------------------------   
    for i in range(3):
        time = data[time_name[i]][key]
        y_sim = data[y_name[i]+"_simulated"][key][:,idx].detach().numpy()

        if i < 2:
            axes.plot(time, 
                      y_sim, 
                      linestyle=linestyle_sim,
                      color=color_sim, 
                      alpha=alpha_sim,
                      linewidth=size_sim)
        else:
            axes.plot(time, 
                      y_sim, 
                      linestyle=linestyle_sim,
                      color=color_sim, 
                      alpha=alpha_sim,
                      linewidth=size_sim,
                      label=label_sim)
        axes.axvline(x=time[-1], color = "grey", linestyle='dashed')
        
    # Set x label
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    
    if type(ylim_up) is not bool:
        axes.set_ylim(top=ylim_up)

    axes.legend()
    
        
    figure_canvas.get_tk_widget().pack(side=tk.TOP, 
                                       fill=tk.BOTH, 
                                       expand=1)
       

           
def config_to_text(config):
    out_text = []
    for key in config.keys():   
        # Write list object in multiple lines             
        if type(config[key]) is list:
            out_text.append(key + ":\n")
            for element in config[key]:
                out_text.append("  - " + str(element) + "\n")
                
        elif type(config[key]) is dict:
            config_key = config[key]
            out_text.append(key + ":\n")
            
            for key in config_key.keys():
                if type(config_key[key]) is list:
                    out_text.append("  " + key + ":\n")
                    for element in config_key[key]:
                        out_text.append("    - " + str(element) + "\n")
                else:
                    out_text.append("  " + key +": " + str(config_key[key]) + "\n")
        else:
            try:
                # Convert time in config to YYYY-MM-DD HH:MM
                if (config[key].shape[0] == 2):
                    out_text.append(key +": \n")
                    if key == "train_period":
                        out_text.append("  - " + str(config["train_period"][0])[:16] + "\n")
                        out_text.append("  - " + str(config["train_period"][1])[:16] + "\n")
                    elif key == "valid_period":
                        out_text.append("  - " + str(config["valid_period"][0])[:16] + "\n")
                        out_text.append("  - " + str(config["valid_period"][1])[:16] + "\n")
                    else:
                        out_text.append("  - " + str(config["test_period"][0])[:16] + "\n")
                        out_text.append("  - " + str(config["test_period"][1])[:16] + "\n")                           
            except:
                # Non list object writte in 1 line
                out_text.append(key +": " + str(config[key]) + "\n")
                #out_text.append("\n")
                
    return out_text
    
def sort_key(config):
    config_sort = {}
        
    if "dynamic_data_file" in config.keys():
        config_sort["dynamic_data_file"] = config["dynamic_data_file"]

    if "static_data_file" in config.keys():
        config_sort["static_data_file"] = config["static_data_file"]

    if "output_directory" in config.keys():
        config_sort["output_directory"] = config["output_directory"] 
        
    if "input_static_features" in config.keys():
        config_sort["input_static_features"] = config["input_static_features"] 

    if "input_dynamic_features" in config.keys():
        config_sort["input_dynamic_features"] = config["input_dynamic_features"]        

    if "target_features" in config.keys():
        config_sort["target_features"] = config["target_features"] 

    if "object_id" in config.keys():
        config_sort["object_id"] = config["object_id"]        

    if "train_period" in config.keys():
        config_sort["train_period"] = config["train_period"]

    if "valid_period" in config.keys():
        config_sort["valid_period"] = config["valid_period"]
        
    if "test_period" in config.keys():
        config_sort["test_period"] = config["test_period"]
        
    if "model_class" in config.keys():
        config_sort["model_class"] = config["model_class"]

    if "Regression" in config.keys():
        config_sort["Regression"] = config["Regression"]
        
    if "Regression" in config.keys():
        config_sort["Regression"] = config["Regression"]

    if "scaler_input_dynamic_features" in config.keys():
        config_sort["scaler_input_dynamic_features"] = config["scaler_input_dynamic_features"]        

    if "scaler_input_static_features" in config.keys():
        config_sort["scaler_input_static_features"] = config["scaler_input_static_features"] 

    if "scaler_target_features" in config.keys():
        config_sort["scaler_target_features"] = config["scaler_target_features"]        

    if "hidden_size" in config.keys():
        config_sort["hidden_size"] = config["hidden_size"] 

    if "num_layers" in config.keys():
        config_sort["num_layers"] = config["num_layers"]        
        
    if "n_epochs" in config.keys():
        config_sort["n_epochs"] = config["n_epochs"] 

    if "learning_rate" in config.keys():
        config_sort["learning_rate"] = config["learning_rate"]        

    if "dropout" in config.keys():
        config_sort["dropout"] = config["dropout"] 

    if "warmup_length" in config.keys():
        config_sort["warmup_length"] = config["warmup_length"]        

    if "optim_method" in config.keys():
        config_sort["optim_method"] = config["optim_method"] 

    if "loss_function" in config.keys():
        config_sort["loss_function"] = config["loss_function"]        

    if "sequence_length" in config.keys():
        config_sort["sequence_length"] = config["sequence_length"]

    if "batch_size" in config.keys():
        config_sort["batch_size"] = config["batch_size"]
        
    if "patience" in config.keys():
        config_sort["patience"] = config["patience"]

    if "static_data_file_forecast" in config.keys():
        config_sort["static_data_file_forecast"] = config["static_data_file_forecast"]        

    if "dynamic_data_file_forecast" in config.keys():
        config_sort["dynamic_data_file_forecast"] = config["dynamic_data_file_forecast"]        
        
    if "forecast_period" in config.keys():
        config_sort["forecast_period"] = config["forecast_period"] 

    if "object_id_forecast" in config.keys():
        config_sort["object_id_forecast"] = config["object_id_forecast"]
            
    return config_sort

def write_yml_file(config, out_file):
    # Convert config to text
    output_text = config_to_text(config=sort_key(config))
        
    # Write config to config file
    with open(out_file, "w") as config_file:
        for line in output_text:
            config_file.write(line)
            
class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self, window_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("600x400")
        self.title(window_name)