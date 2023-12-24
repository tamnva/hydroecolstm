
import customtkinter as ctk
import tkcalendar as tkc
from hydroecolstm.data.read_data import read_forecast_data
import tkinter as tk
import numpy as np
from CTkListbox import CTkListbox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class ForecastFrame(ctk.CTkFrame):
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
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5)
        self.tabview.grid(row=0, column=0, padx = 10, pady=(5,5), sticky="we")  
        self.tabview.add("1. Inputs")
        self.tabview.tab("1. Inputs").grid_columnconfigure(0, weight=0) 
        self.tabview.tab("1. Inputs").grid_columnconfigure(1, weight=0)
        self.tabview.tab("1. Inputs").grid_columnconfigure(2, weight=1)

        
        self.tabview.add("2. Outputs")
        self.tabview.tab("2. Outputs").grid_columnconfigure((0), weight=1)
        
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

        # -----------------------------------------------------------Column 2
        self.file_selection_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="2. Input data files")
        self.file_selection_label.grid(row=0, column=1, padx = 10, pady=(5,5), sticky="w")

        self.load_train_test_config = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Select train/test config file", 
                                               command=None)
        self.load_train_test_config.grid(row=1, column=1, padx = 10, pady=(2,2), sticky="w")  
        
        self.dynamic_file_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Select dynamic data file", 
                                               command=None)
        self.dynamic_file_button.grid(row=2, column=1, padx = 10, pady=(2,2), sticky="w")       
    
        self.static_file_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Select static data file", 
                                               command=None)
        self.static_file_button.grid(row=3, column=1, padx = 10, pady=(2,2), sticky="w")  

        self.model_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Load trained model", 
                                               command=None)
        self.model_button.grid(row=4, column=1, padx = 10, pady=(2,2), sticky="w")    
        
        self.model_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="3. Forecast period")
        self.model_label.grid(row=5, column=1, padx = 10, pady=(40,5), sticky="w")
        
        self.start_forecast = tkc.DateEntry(self.tabview.tab("1. Inputs"), 
                                         date_pattern= 'yyyy-mm-dd', width = 25,
                                         year=1800, month=1, day=1, font=ctk.CTkFont(size=14))
        self.start_forecast.grid(row= 6, column=1, padx=30, pady=10, sticky='e')
        self.end_forecast = tkc.DateEntry(self.tabview.tab("1. Inputs"), 
                                       date_pattern= 'yyyy-mm-dd', width = 25,
                                       year=2015, month=1, day=1, font=ctk.CTkFont(size=14))
        self.end_forecast.grid(row= 7, column=1, padx=30, pady=10, sticky='e')   
        
        # ---------------------------------------------------------------Column 3
        self.object_id_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                 text="4. Select object_id for forecasting")
        self.object_id_label.grid(row=0, column=3, columnspan=1, padx = 10, pady=(5,5), sticky="w")
        self.object_id_forecast = CTkListbox(master=self.tabview.tab("1. Inputs"), 
                                           multiple_selection=True, border_width=1.5,
                                           text_color="black")
        
        self.object_id_forecast.grid(row=1, column=3, rowspan=4, padx = 10, pady=(10,10), sticky="we")

        self.run_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                        text="5. Run forecast")
        self.run_label.grid(row=5, column=3, padx = 10, pady=(5,5), sticky="w")
       
        self.run_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                              anchor='w', 
                                              text="Run", 
                                              command=None)
        self.run_button.grid(row=6, column=3, padx = 10, pady=(5,5), sticky="w") 
        
        #---------------------------------------------------------2 Outputs
        self.object_id_label = ctk.CTkLabel(self.tabview.tab("2. Outputs"), 
                                            text="1. Please insert object_id for plot")
        self.object_id_label.grid(row=0, column=0, sticky="w", padx=(5,5))
        
        self.select_input_frame = ctk.CTkFrame(master=self.tabview.tab("2. Outputs"), height=400)
        self.select_input_frame.grid(row=1, column=0, sticky="w", padx=(20,20), pady=(20,20))
        self.select_input_frame.columnconfigure((0,1), weight=1)

        self.object_id_label = ctk.CTkLabel(self.tabview.tab("2. Outputs"), 
                                            text="2. Plotting area")
        self.object_id_label.grid(row=2, column=0, sticky="w", padx=(5,5), pady=(20,5))
        
        self.plot_frame = ctk.CTkCanvas(master=self.tabview.tab("2. Outputs"), height=400)
        self.plot_frame.grid(row=3, column=0, sticky="w", padx=(20,20), pady=(20,20))  

        self.object_id = ctk.CTkTextbox(master=self.select_input_frame, height=30)
        self.object_id.insert("0.0", "object_id") 
        self.object_id.grid(row=0, column=0, sticky="w", padx=(5,5), pady=(5,5))
        self.object_id.bind('<KeyRelease>', self.get_object_id)

        self.target_feature = ctk.CTkTextbox(master=self.select_input_frame, height=30)
        self.target_feature.insert("0.0", "target_feature") 
        self.target_feature.grid(row=1, column=0, sticky="w", padx=(5,5), pady=(5,5))
        self.target_feature.bind('<KeyRelease>', self.get_target_feature)
        
        self.next_object_id_button = ctk.CTkButton(self.select_input_frame, 
                                                   anchor='w',
                                                   command=self.next_object_id, 
                                                   text="Next object id")
        self.next_object_id_button.grid(row=0, column=1, sticky="w", 
                                        padx=(5,5), pady=(5,5))
        self.next_target_feature_button = ctk.CTkButton(self.select_input_frame, 
                                                        anchor='w',
                                                   command=self.next_target_feature, 
                                                   text="Next target features")
        self.next_target_feature_button.grid(row=1, column=1, sticky="w", 
                                             padx=(5,5), pady=(5,5))
                
        self.update_plot = ctk.CTkButton(self.select_input_frame, anchor='w', 
                                 command=self.plot_figure, text="Update plot")
        self.update_plot.grid(row=3, column=0, columnspan=2, sticky="w", 
                              padx=(5,20), pady=(10,10))

    # Get dropout
    
    def import_button_event(self):
        
        if self.import_checkbox.get() == 1:
            
            # Deactive other buttons
            self.load_train_test_config.configure(state="disabled")
            self.dynamic_file_button.configure(state="disabled")
            self.static_file_button.configure(state="disabled")
            self.model_button.configure(state="disabled")
            
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
                self.config["dynamic_data_file_forecast"] = "dynamic_data_file"
            if "static_data_file" in self.config.keys():
                self.config["static_data_file"] = "static_data_file"
                
        else:
            # Activate other buttons
            self.load_train_test_config.configure(state="normal")
            self.dynamic_file_button.configure(state="normal")
            self.static_file_button.configure(state="normal")
            self.model_button.configure(state="normal")
                
        
    def run_forecast(self):
        
        try:
            # Get forecast period
            self.config['forecast_period'] = [self.start_forecast.get_date(),
                                              self.start_forecast.get_date()]
            
            # Get forecast id
            all_items = self.object_id_forecast.get(index='all')
            select_index = self.object_id_forecast.curselection()
            object_id_forecast = [all_items[i] for i in select_index]
            self.config['object_id_forecast'] = object_id_forecast
            
            print(self.config['object_id_forecast'])
            print(self.config['forecast_period'] )
            
            # Get forecast data
            predict_data = read_forecast_data(self.config)           
            self.globalData["x_forecast"] = predict_data["x_forecast"]
            self.globalData["y_forecast"] = predict_data["y_forecast"]
            self.globalData["time_forecast"] = predict_data["time_forecast"]
            self.globalData["x_forecast_column_name"] = predict_data["x_column_name"]
            self.globalData["y_forecast_column_name"] = predict_data["y_column_name"]         
            del predict_data
            
            # Scale forecast data
            self.globalData["x_forecast_scale"] =\
                self.globalData["x_scaler"].transform(x=self.globalData["x_forecast"])

            self.globalData["y_forecast_scale"] =\
                self.globalData["y_scaler"].transform(x=self.globalData["y_forecast"])

            # Run forward model
            self.globalData["y_forecast_scale_predict"] =\
                self.globalData["model"].forward(self.globalData["x_forecast_scale"])
            
            # Scale forecast data
            print(self.globalData["y_forecast_scale_predict"])
   
        except:
            pass
        # Read and split data forecast 
        #

    # Get dropout
    def next_object_id(self):
        try:
            if self.globalData["object_id_no"] > len(self.config["object_id"]) - 1:
                self.globalData["object_id_no"] = 0
            
            #print(self.globalData["object_id_no"])
            self.object_id.delete("0.0", "end")
            self.object_id.insert("0.0", self.config["object_id"]
                                  [self.globalData["object_id_no"]])
            self.globalData["object_id_plot"] = str(self.config["object_id"]
                                                    [self.globalData["object_id_no"]])
                
            #print(self.globalData["object_id_plot"])
                
            self.globalData["object_id_no"] += 1
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
        
        # Remove and create frame again to update figure
        self.plot_frame.destroy()
        self.plot_frame = ctk.CTkFrame(master=self, height=400)
        self.plot_frame.grid(row=3, column=0, sticky="w", padx=(20,20), pady=(20,20))
        
        try:
            time = self.globalData["time_forecast"][self.globalData["object_id_forecast_plot"]]
            
            obs = self.globalData["y_forecast_scale"][self.globalData["object_id_forecast_plot"]]  
            obs = obs[:, self.config["target_features"].\
                      index(self.globalData["target_feature_forecast_plot"])]  
                
            predict = self.globalData["y_forecast_scale_predict"]\
                [self.globalData["object_id_forecast_plot"]].detach().numpy()
            predict = predict[:, self.config["target_features"].\
                              index(self.globalData["target_feature_forecast_plot"])]    
            
            figure = Figure(figsize=(15, 4), dpi=100)
            figure_canvas = FigureCanvasTkAgg(figure, self.plot_frame )
            NavigationToolbar2Tk(figure_canvas, self.plot_frame )          
            axes = figure.add_subplot()
            axes.plot(time, obs, 'ro', label = "Observed (test data)", 
                      alpha=0.9, markersize=2.5 )            
            axes.plot(time, predict, color = 'blue', label = "Predicted (test data)", 
                      alpha=0.9, linewidth=0.75)
            axes.set_title(f"object_id = {self.globalData['object_id_forecast_plot']}")
            axes.legend() 
            figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
        except:
            x = 4 + np.random.normal(0, 2, 24)
            y = 4 + np.random.normal(0, 2, len(x))
            figure = Figure(figsize=(15, 4), dpi=100)
            figure_canvas = FigureCanvasTkAgg(figure, self.plot_frame )
            NavigationToolbar2Tk(figure_canvas, self.plot_frame )
            axes = figure.add_subplot()            
            axes.plot(x, color = 'blue', label = "Predicted (test data)", 
                      alpha=0.9, linewidth=0.75)
            axes.plot(y, 'ro', label = "Observed (test data)", 
                      alpha=0.9, markersize=2.5 )
            axes.legend()
            axes.set_title("Test plot")
                
            figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1) 
