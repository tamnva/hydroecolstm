import matplotlib
import customtkinter as ctk
import pandas as pd
import tkcalendar as tkc
from hydroecolstm.data.read_data import read_forecast_data
import tkinter as tk
import torch
from CTkListbox import CTkListbox
from CTkToolTip import CTkToolTip
from pandastable import Table
from hydroecolstm.data.read_config import read_config
from hydroecolstm.model.lstm_linears import Lstm_Linears
from hydroecolstm.model.ea_lstm import Ea_Lstm_Linears
from hydroecolstm.interface.utility import (ToplevelWindow, 
                                            plot_train_valid_loss,
                                            plot_time_series,
                                            check_size,check_linestyle,
                                            check_alpha,check_line_plot,
                                            check_marker, check_color, 
                                            check_ylim, combine_forecast)

class ApplicationFrame(ctk.CTkScrollableFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config = config
        self.globalData = globalData
        self.columnconfigure((0), weight=1)
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self): 
        # ---------------------------------------------
        # create tabs
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5, 
                                      fg_color = "transparent")
        self.tabview.pack(fill='both',expand=1)
        self.tabview.add("1. Inputs")
        self.tabview.tab("1. Inputs").grid_columnconfigure(0, weight=0) 
        self.tabview.tab("1. Inputs").grid_columnconfigure(1, weight=0)
        self.tabview.tab("1. Inputs").grid_columnconfigure(2, weight=0)
        self.tabview.tab("1. Inputs").grid_columnconfigure(3, weight=1)

        
        self.tabview.add("2. Plot outputs")
        self.tabview.tab("2. Plot outputs").grid_columnconfigure((0,1,2), weight=1)
        
        # ---------------------------------------------content of load data tab
        # -----------------------------------------------------------Column 1
        self.import_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="1. Internal import")
        self.import_label.grid(row=0, column=0, padx = 10, pady=(5,5), sticky="w")
        self.import_checkvalue = ctk.IntVar(value=0)         
        self.import_checkbox = ctk.CTkCheckBox(self.tabview.tab("1. Inputs"), 
                                               text=" ",
                                               command=self.import_button_event, 
                                               variable=self.import_checkvalue)
        self.import_checkbox.grid(row=1, column=0, padx = 10, pady=(10,10), sticky="we")
        CTkToolTip(self.import_checkbox, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="Check this box to use the model and data created from steps 1 to 4")

        # -----------------------------------------------------------Column 2
        self.file_selection_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="2. Input data files")
        self.file_selection_label.grid(row=0, column=1, padx = 10, pady=(5,5), sticky="w")

        self.load_train_test_config = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Select config file", 
                                               command=self.get_config_file)
        
        CTkToolTip(self.load_train_test_config, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="Select the configuration .yml file")
        
        self.load_train_test_config.grid(row=1, column=1, padx = 10, pady=(2,2), sticky="w")  
        self.config_file_name = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="No file was selected")
        self.config_file_name.grid(row=1, column=2, padx = 10, pady=(2,2), sticky="w")  
        
        self.dynamic_file_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Select dynamic data file", 
                                               command=self.get_dynamic_file_forecast)
        self.dynamic_file_button.grid(row=2, column=1, padx = 10, pady=(2,2), sticky="w")
        CTkToolTip(self.dynamic_file_button, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="Please see 1. Data processing for more detail")
        
        self.dynamic_file_name = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="No file was selected")
        self.dynamic_file_name.grid(row=2, column=2, padx = 10, pady=(2,2), sticky="w") 
        
    
        self.static_file_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Select static data file", 
                                               command=self.get_static_file_forecast)
        self.static_file_button.grid(row=3, column=1, padx = 10, pady=(2,2), sticky="w")
        CTkToolTip(self.static_file_button, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="Please see 1. Data processing for more detail")
        
        self.static_file_name = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="No file was selected")
        self.static_file_name.grid(row=3, column=2, padx = 10, pady=(2,2), sticky="w") 
        

        self.model_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Load trained model", 
                                               command=self.load_model_state_dicts)
        self.model_button.grid(row=4, column=1, padx = 10, pady=(2,2), sticky="w") 
        CTkToolTip(self.static_file_button, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="Select file contains the model state dicts .pt file")
        
        self.model_name = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="No model was loaded")
        self.model_name.grid(row=4, column=2, padx = 10, pady=(2,2), sticky="w") 
        
        self.load_scaler_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Load scalers", 
                                               command=self.load_scalers)
        self.load_scaler_button.grid(row=5, column=1, padx = 10, pady=(2,2), sticky="w") 
        self.load_scaler_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="No scalers were loaded")
        self.load_scaler_label.grid(row=5, column=2, padx = 10, pady=(2,2), sticky="w") 
        CTkToolTip(self.static_file_button, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="Load the file contains scaler for input and target features. " +  
                   " When you save this project, this is also inside the data.pt file." +
                   " So you can load this data.pt file here")
        
        
        self.model_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="3. Forecast period")
        self.model_label.grid(row=6, column=1, columnspan=2, padx = 10, pady=(40,5), sticky="w")
        self.start_forecast = tkc.DateEntry(self.tabview.tab("1. Inputs"), 
                                         date_pattern= 'yyyy-mm-dd', width = 25,
                                         year=1800, month=1, day=1, font=ctk.CTkFont(size=14))
        self.start_forecast.grid(row= 7, column=1, columnspan=2, padx=30, pady=10, sticky='w')
        self.end_forecast = tkc.DateEntry(self.tabview.tab("1. Inputs"), 
                                       date_pattern= 'yyyy-mm-dd', width = 25,
                                       year=2015, month=1, day=1, font=ctk.CTkFont(size=14))
        self.end_forecast.grid(row= 8, column=1, columnspan=2, padx=30, pady=10, sticky='w')   
        
        
        # ---------------------------------------------------------------Column 3
        self.object_id_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                 text="4. Select object_id for forecasting")
        self.object_id_label.grid(row=0, column=3, columnspan=1, 
                                  padx = 10, pady=(5,5), sticky="w")
        self.object_id_forecast = CTkListbox(master=self.tabview.tab("1. Inputs"), 
                                           multiple_selection=True, border_width=1.5,
                                           text_color="black")
        
        self.object_id_forecast.grid(row=1, column=3, rowspan=5, padx = 10, 
                                     pady=(10,10), sticky="we")

        self.run_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                        text="5. Run forecast")
        self.run_label.grid(row=6, column=3, padx = 10, pady=(5,5), sticky="w")
       
        self.run_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                              anchor='w', 
                                              text="Run", 
                                              command=self.run_forecast)
        self.run_button.grid(row=7, column=3, padx = 10, pady=(5,5), sticky="w") 
        
        #---------------------------------------------------------2 Plot output
        self.object_id_label =\
            ctk.CTkLabel(self.tabview.tab("2. Plot outputs"),
                         text="1. Select object id and target feature for plot")
            
        self.object_id_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=5)

        self.object_id=ctk.CTkTextbox(self.tabview.tab("2. Plot outputs"), 
                                        height=30, border_width=1.5)
        
        self.object_id.insert("0.0", "object_id") 
        self.object_id.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.object_id.bind('<KeyRelease>', self.get_object_id)

        self.target_feature=ctk.CTkTextbox(self.tabview.tab("2. Plot outputs"), 
                                             height=30, border_width=1.5)
        
        self.target_feature.insert("0.0", "target_feature") 
        
        self.target_feature.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        self.target_feature.bind('<KeyRelease>', self.get_target_feature)
        
        self.next_object_id_button=ctk.CTkButton(self.tabview.tab("2. Plot outputs"), 
                                                   anchor='w',
                                                   command=self.next_object_id, 
                                                   text="Next object id")
        
        self.next_object_id_button.grid(row=1, column=1, sticky="w", 
                                        padx=5, pady=5)
        
        self.next_target_feature_button =\
            ctk.CTkButton(self.tabview.tab("2. Plot outputs"), anchor='w',
                          command=self.next_target_feature,
                          text="Next target features")
            
        self.next_target_feature_button.grid(row=2, column=1, sticky="w", 
                                             padx=5, pady=5)
        
        # Plot button
        self.update_plot=ctk.CTkButton(self.tabview.tab("2. Plot outputs"), anchor='we', 
                                 command=self.plot_figure, 
                                 text="Plot (update plot)")
        
        self.update_plot.grid(row=1, column=2,  
                              sticky="we", padx=5, pady=5)
        
        # show all data
        self.show_all_data=ctk.CTkButton(self.tabview.tab("2. Plot outputs"), anchor='w', 
                                         command=self.show_all_data_event,
                                         text="Show data source")
        self.show_all_data.grid(row=2, column=2, sticky="we", padx=5, pady=5)
        CTkToolTip(self.show_all_data, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, justify="left", 
                   message='Click here to show all data')
        
        
        #-------------------------------------------------------2. Plot setting
        # check to load/unload
        self.plot_timeseries=ctk.IntVar(value=0)        
        self.plot_timeseries_checkbox=ctk.CTkCheckBox(self.tabview.tab("2. Plot outputs"), 
                                               text="Show plot settings",
                                               command= self.show_plot_timeseries_setting, 
                                               variable=self.plot_timeseries)
        
        self.plot_timeseries_checkbox.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")
        
        self.plot_timeseries_frame=ctk.CTkFrame(self.tabview.tab("2. Plot outputs"),
                                              height=400, fg_color="transparent")
        self.plot_timeseries_frame.columnconfigure((0,1,2), weight=1)

        #----------------------------------------------------------------------
        self.common_settings=ctk.CTkLabel(self.plot_timeseries_frame,
                                         text="Common settings")   
        self.common_settings.grid(row=0, column=0, sticky="w", padx=5)
        
        self.plot_title=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="plot title")
        
        self.plot_title.grid(row=1, column=0, sticky="w", padx=5)
        
        
        self.x_label=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="xlabel")
        self.x_label.grid(row=2, column=0, sticky="w", padx=5)
        
        self.y_label=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="ylabel")
        self.y_label.grid(row=3, column=0, sticky="w", padx=5)
        
        self.ylim_up=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="ylim (upper)")
        CTkToolTip(self.ylim_up, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, justify="left",
                   message='Upper limit of the y axis (input numeric value)')
        
        self.ylim_up.grid(row=4, column=0, sticky="w", padx=5)
        
        self.observed_data_plot=ctk.CTkLabel(self.plot_timeseries_frame,
                                         text="observed data plot")
        self.observed_data_plot.grid(row=0, column=1, sticky="w", padx=5)
          
        self.lineplot=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="line (true or false)")
        self.lineplot.grid(row=1, column=1, sticky="w", padx=5, pady=5)
                
        self.color_obs=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="color")
        self.color_obs.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        CTkToolTip(self.color_obs, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, 
                   justify="left", 
                   message=str(list(matplotlib.colors.cnames.keys())))
        
        self.alpha_obs=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="alpha")    
        self.alpha_obs.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        CTkToolTip(self.alpha_obs, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, 
                   justify="left", message='tranparency level: input any ' +
                   'numeric values between 0 and 1')
        
        self.size_obs=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="size")        
        self.size_obs.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        CTkToolTip(self.size_obs, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, justify="left", 
                   message='input positive numeric value for line thickness')
        
        self.linestyle_obs=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="linestyle")    
        CTkToolTip(self.linestyle_obs, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, 
                   justify="left", message='solid, dashed, dashdot, dotted')
        
        self.linestyle_obs.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        self.marker_obs=ctk.CTkEntry(self.plot_timeseries_frame,
                                       placeholder_text="marker")
        self.marker_obs.grid(row=6, column=1, sticky="w", padx=5, pady=5)
        CTkToolTip(self.marker_obs, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, 
                   justify="left", message="., o, s, ^, v, +, x")
        
        self.label_obs=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="label")
        
        self.label_obs.grid(row=7, column=1, sticky="w", padx=5, pady=5)
        
        #----------------------------------------------------------------------
        self.simulated_data_plot=ctk.CTkLabel(self.plot_timeseries_frame,
                                         text="simulated data plot")
        self.simulated_data_plot.grid(row=0, column=3, sticky="w", padx=5)
        
        self.color_sim=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="color")
        self.color_sim.grid(row=2, column=3, sticky="w", padx=5)
        self.alpha_sim=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="alpha")
        self.alpha_sim.grid(row=3, column=3, sticky="w", padx=5)
        self.size_sim=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="size")
        self.size_sim.grid(row=4, column=3, sticky="w", padx=5)
        CTkToolTip(self.size_sim, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, justify="left", 
                   message='input positive numeric value for line thickness')
        
        self.linestyle_sim=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="linestype")
        self.linestyle_sim.grid(row=5, column=3, sticky="w", padx=5)
        CTkToolTip(self.linestyle_sim, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, 
                   justify="left", message='solid, dashed, dashdot, dotted')
        self.label_sim=ctk.CTkEntry(self.plot_timeseries_frame,
                                  placeholder_text="label")
        self.label_sim.grid(row=7, column=3, sticky="w", padx=5)

    def show_all_data_event(self):
        try:
            dataframe = combine_forecast(self.globalData,
                                         self.globalData["y_column_name"])
            
            self.table = Table(tk.Toplevel(self), dataframe=dataframe, 
                               showtoolbar=True, showstatusbar=True)
            self.table.show()
        except:
            pass
        
    def get_config_file(self):
        
        try:
            # get file name
            file_name = ctk.filedialog.askopenfilename(title="Select configuration file (.yml format)", 
                                                       filetypes=(('yml files', '*.yml'),
                                                                  ('All files', '*.*')))
            
            # read configuration file
            config = read_config(file_name)
            
            # update label
            self.config_file_name.configure(text= '...' + file_name[-15:])
            
            # update configuration
            self.config = config
            
            tk.messagebox.showinfo(title="Message box", 
                                   message="All of your previous configurations " +
                                   "were replaced by the configurations in the " +
                                   "selected file")
            
        except:
            tk.messagebox.showinfo(title="Message box", 
                                   message="No thing was change due either " +
                                   "canceling file selection or unsuccessfully " +
                                   "read of the selected configuration file")
            
    def get_dynamic_file_forecast(self):
        
        try:
            # get file name
            file_name = ctk.filedialog.askopenfilename(title="Select dynamic data file (.csv format)", 
                                                       filetypes=(('yml files', '*.csv'),
                                                                  ('All files', '*.*')))
            
            # update label
            self.dynamic_file_name.configure(text= '...' + file_name[-15:])
            
            # update configuration
            self.config["dynamic_data_file_forecast"] = [file_name]
            
            # try to remove object_id
            try:
                self.object_id_forecast.delete(index=0, last='END')
            except:
                pass
            
            # read dynamic data file to get header namey (object_id)
            object_id = pd.read_csv(file_name, delimiter=",", header=0)
            object_id = list(pd.unique(object_id["object_id"]))
             
            # remove columns with object_id and time
            for i in object_id:
                self.object_id_forecast.insert('END', option=i)
            
        except:
            self.dynamic_file_name.configure(text= 'No file was selected')


    def get_static_file_forecast(self):
        
        try:
            # get file name
            file_name = ctk.filedialog.askopenfilename(title="Select static data file (.csv format)", 
                                                       filetypes=(('yml files', '*.csv'),
                                                                  ('All files', '*.*')))
            
            # update label
            self.static_file_name.configure(text= '...' + file_name[-15:])
            
            # update configuration
            self.config["static_data_file_forecast"] = [file_name]
            
            
        except:
            self.static_file_name.configure(text= 'No file was selected')
            
    def load_model_state_dicts(self):
        try:
            # Create the model
            if self.config["model_class"] == "LSTM":
                self.globalData["model"] = Lstm_Linears(self.config)
            else:
                self.globalData["model"] = Ea_Lstm_Linears(self.config)
                
            # Load model state dicts
            file_name = ctk.filedialog.askopenfilename(title="Select model state dicts (.pt format)", 
                                                       filetypes=(('yml files', '*.pt'),
                                                                  ('All files', '*.*')))
            self.globalData["model"].load_state_dict(torch.load(file_name))
            
            # Set model to eval mode (in this mode, dropout = 0, no normlization)
            self.globalData["model"].eval()
            
            # Update selected model name
            self.model_name.configure(text= '...' + file_name[-15:])
            
        except:
            tk.messagebox.showinfo(title="Error", 
                                   message="Cannot load model state dicts")
            
    # Load scalers
    def load_scalers(self):
        try:
            # Load model state dicts
            file_name = ctk.filedialog.askopenfilename(title="Select scaler file (.pt format)", 
                                                       filetypes=(('yml files', '*.pt'),
                                                                  ('All files', '*.*')))
            
            scalers = torch.load(file_name)
            self.globalData["x_scaler"] = scalers["x_scaler"]
            self.globalData["y_scaler"] = scalers["y_scaler"]
            
            # Update selected model name
            self.load_scaler_label.configure(text= '...' + file_name[-15:])
            
        except:
            tk.messagebox.showinfo(title="Error", 
                                   message="Cannot load scalers")        
        
    # Show plot time series setting
    def show_plot_timeseries_setting(self):
        if self.plot_timeseries.get() == 0:
            self.plot_timeseries_frame.grid_forget()
        else:
            self.plot_timeseries_frame.grid(row=4, column=0, columnspan=3,
                                      sticky="w", padx=(0,0), pady=(20,20))
            
    # Get dropout
    def import_button_event(self):
        
        if self.import_checkbox.get() == 1:
            
            # Deactive other buttons
            self.load_train_test_config.configure(state="disabled")
            self.dynamic_file_button.configure(state="disabled")
            self.static_file_button.configure(state="disabled")
            self.model_button.configure(state="disabled")
            self.load_scaler_button.configure(state="disabled")
            
            # First try to remove all items in object_id_forecast
            try:
                self.object_id_forecast.delete(index=0, last='END')
            except:
                pass
            
            # Add items to object_id_forecast from globalData
            try:
                object_id = self.globalData['object_id'].copy()
                for i in object_id:
                    self.object_id_forecast.insert('END', option=i)
            except:
                pass
            
            # Input dynamic and static data
            if "dynamic_data_file" in self.config.keys():
                self.config["dynamic_data_file_forecast"] = ["dynamic_data_file"]
            if "static_data_file" in self.config.keys():
                self.config["static_data_file_forecast"] = ["static_data_file"]
                
        else:
            # Activate other buttons
            self.load_train_test_config.configure(state="normal")
            self.dynamic_file_button.configure(state="normal")
            self.static_file_button.configure(state="normal")
            self.model_button.configure(state="normal")
            self.load_scaler_button.configure(state="normal")
            
                
        
    def run_forecast(self):
        
        try:
            
            
            # Get forecast period
            self.config['forecast_period'] = pd.to_datetime(
                [self.start_forecast.get_date(), self.end_forecast.get_date()],
                format = "%Y-%m-%d %H:%M")

            # Get forecast id
            all_items = self.object_id_forecast.get(index='all')
            select_index = self.object_id_forecast.curselection()
            object_id_forecast = [all_items[i] for i in select_index]
            self.config['object_id_forecast'] = object_id_forecast

            # Add forecast data to globalData
            self.globalData.update(read_forecast_data(self.config))
 
            # Scale forecast data
            self.globalData["x_forecast_scale"] =\
                self.globalData["x_scaler"].transform(x=self.globalData["x_forecast"])

            # Run forward model
            y_forecast_scale_simulated =\
                self.globalData["model"].evaluate(self.globalData["x_forecast_scale"])

            tk.messagebox.showinfo(title="Message box", 
                                   message="Finished forward run")
            
            # Inverse transform y forcast to original scale
            self.globalData["y_forecast_simulated"] =\
                self.globalData["y_scaler"].inverse(y_forecast_scale_simulated)
            
        except:
            tk.messagebox.showinfo(title="Message box", 
                                   message="Error: Cannot run with forecast data")

    # Get dropout
    def next_object_id(self):
        try:
            if self.globalData["object_id_forecast_no"] > len(self.config["object_id_forecast"]) - 1:
                self.globalData["object_id_forecast_no"] = 0
            
            #print(self.globalData["object_id_forecast_no"])
            self.object_id.delete("0.0", "end")
            self.object_id.insert("0.0", self.config["object_id_forecast"]
                                  [self.globalData["object_id_forecast_no"]])
            self.globalData["object_id_forecast_plot"] = str(self.config["object_id_forecast"]
                                                    [self.globalData["object_id_forecast_no"]])
                
            self.globalData["object_id_forecast_no"] += 1
        except:
            None

    def next_target_feature(self):
        try:
            if self.globalData["target_feature_forecast_no"] >\
                len(self.config["target_features"]) - 1:
                self.globalData["target_feature_forecast_no"] = 0
               
            self.target_feature.delete("0.0", "end")
            
            self.target_feature.insert("0.0", self.config["target_features"]
                                       [self.globalData["target_feature_forecast_no"]])
            
            self.globalData["target_feature_forecast_plot"] =\
                str(self.config["target_features"]\
                    [self.globalData["target_feature_forecast_no"]])
                
            self.globalData["target_feature_forecast_no"] += 1
        except:
            None
            
    def get_object_id(self, dummy):
        self.globalData["object_id_forecast_plot"] =\
            str(self.object_id.get("0.0", "end"))
            
        self.globalData["object_id_forecast_plot"] =\
            self.globalData["object_id_forecast_plot"].strip()
            
        print(f"Selected object_id for plot =\
              {self.globalData['object_id_forecast_plot']}")

    def get_target_feature(self, dummy):
        self.globalData["target_feature_forecast_plot"] =\
            str(self.target_feature.get("0.0", "end"))
        self.globalData["target_feature_forecast_plot"] =\
            self.globalData["target_feature_forecast_plot"].strip()
        print(f"Selected target_feature for plot =\
              {self.globalData['target_feature_forecast_plot']}")
              
    def plot_figure(self):
        try:
            # Get information for plot
            key=self.globalData["object_id_forecast_plot"]
            idx=self.config["target_features"].index(
                self.globalData["target_feature_forecast_plot"])
            
            # General setting for plotting observed data
            lineplot=check_line_plot(self.lineplot)
            color_obs=check_color(self.color_obs, "coral")
            alpha_obs=check_alpha(self.alpha_obs, 1)
            size_obs=check_size(self.size_obs, 1)
            linestyle_obs=check_linestyle(self.linestyle_obs, 'dashed')
            marker_obs=check_marker(self.marker_obs, "d")
            label_obs=self.label_obs.get()
            if len(label_obs) == 0: label_obs="Observed"

            # General setting for plotting simulated data
            color_sim=check_color(self.color_sim, "blue")
            alpha_sim=check_alpha(self.alpha_sim, 1)
            size_sim=check_size(self.size_sim, 1)
            linestyle_sim=check_linestyle(self.linestyle_sim, 'solid')
            label_sim=self.label_sim.get()
            if len(label_sim) == 0: label_sim="Simulated"
          
            # Lable
            title=self.plot_title.get()
            xlabel=self.x_label.get()
            ylabel=self.y_label.get().strip()
            if ylabel == "": ylabel=self.globalData["target_feature_forecast_plot"]
            ylim_up=check_ylim(self.ylim_up, -9999.0)
                
            plot_window=ToplevelWindow(window_name="Plot window")
            
            # Plot
            plot_time_series(plot_window, self.globalData, key, idx, lineplot, 
                             color_obs, alpha_obs, size_obs, linestyle_obs,
                             marker_obs, label_obs, color_sim, alpha_sim, 
                             size_sim, linestyle_sim, label_sim, title, xlabel,
                             ylabel, ylim_up, forecast_period=True)
            
            
        except:
            tk.messagebox.showinfo(title="Error", 
                                   message="Cannot plot time series")
              
              

