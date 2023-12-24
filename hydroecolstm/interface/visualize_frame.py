
import customtkinter as ctk
import tkinter as tk
import numpy as np
from pathlib import Path
import torch
import yaml
import pickle 

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



class VisualizeFrame(ctk.CTkFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config = config
        self.globalData = globalData
        self.columnconfigure((0), weight=1)
        #self.rowconfigure((0,1), weight=1)
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self): 
        # ---------------------------------------------content of load data tab
        self.object_id_label = ctk.CTkLabel(self, text="1. Select object id and target feature for plot")
        self.object_id_label.grid(row=0, column=0, sticky="w", padx=(5,5))
        self.select_input_frame = ctk.CTkFrame(master=self, height=400)
        self.select_input_frame.grid(row=1, column=0, sticky="w", padx=(20,20), pady=(20,20))
        self.select_input_frame.columnconfigure((0,1,3,4), weight=0)

        self.object_id_label = ctk.CTkLabel(self, text="2. Plotting area")
        self.object_id_label.grid(row=2, column=0, sticky="w", padx=(5,5))
        self.plot_frame = ctk.CTkCanvas(master=self, height=400)
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
        
        
        self.update_plot = ctk.CTkButton(self.select_input_frame, anchor='we', 
                                 command=self.plot_figure, text="Update plot")
        self.update_plot.grid(row=3, column=0, columnspan=2, sticky="we", padx=(5,5), pady=(20,20))

        self.save_project = ctk.CTkButton(self.select_input_frame, anchor='w',
                                          command=self.save_project_event,
                                          text="Save project") 
        self.save_project.grid(row=3, column=2, sticky="e", padx=(5,5), pady=(20,20))
        self.load_project = ctk.CTkButton(self.select_input_frame, anchor='w',
                                          command=self.load_project_event, 
                                          text="Load project") 
        self.load_project.grid(row=3, column=3, sticky="e", padx=(5,5), pady=(20,20))
    
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
            if self.globalData["target_feature_no"] > len(self.config["target_features"]) - 1:
                self.globalData["target_feature_no"] = 0
               
            self.target_feature.delete("0.0", "end")
            self.target_feature.insert("0.0", self.config["target_features"]
                                       [self.globalData["target_feature_no"]])
            self.globalData["target_feature_plot"] = str(self.config["target_features"]
                                                         [self.globalData["target_feature_no"]])
            self.globalData["target_feature_no"] += 1
        except:
            None
            
    def get_object_id(self, dummy):
        self.globalData["object_id_plot"] = str(self.object_id.get("0.0", "end"))
        self.globalData["object_id_plot"] = self.globalData["object_id_plot"].strip()
        print(f"Selected object_id for plot = {self.globalData['object_id_plot']}")

    def get_target_feature(self, dummy):
        self.globalData["target_feature_plot"] = str(self.target_feature.get("0.0", "end"))
        self.globalData["target_feature_plot"] = self.globalData["target_feature_plot"].strip()
        print(f"Selected target_feature for plot = {self.globalData['target_feature_plot']}")
       
    def plot_figure(self):
        
        # Remove and create frame again to update figure
        self.plot_frame.destroy()
        self.plot_frame = ctk.CTkFrame(master=self, height=400)
        self.plot_frame.grid(row=3, column=0, sticky="w", padx=(20,20), pady=(20,20))
        
        try:
            time = self.globalData["time_test"][self.globalData["object_id_plot"]]
            
            obs = self.globalData["y_test_scale"][self.globalData["object_id_plot"]]  
            obs = obs[:, self.config["target_features"].\
                      index(self.globalData["target_feature_plot"])]  
                
            predict = self.globalData["y_test_scale_predict"]\
                [self.globalData["object_id_plot"]].detach().numpy()
            predict = predict[:, self.config["target_features"].\
                              index(self.globalData["target_feature_plot"])]    
            
            figure = Figure(figsize=(15, 4), dpi=100)
            figure_canvas = FigureCanvasTkAgg(figure, self.plot_frame )
            NavigationToolbar2Tk(figure_canvas, self.plot_frame )          
            axes = figure.add_subplot()
            axes.plot(time, obs, 'ro', label = "Observed (test data)", 
                      alpha=0.9, markersize=2.5 )            
            axes.plot(time, predict, color = 'blue', label = "Predicted (test data)", 
                      alpha=0.9, linewidth=0.75)
            axes.set_title(f"object_id = {self.globalData['object_id_plot']}")
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
            
    def save_project_event(self):    
        
        output_dir = tk.filedialog.askdirectory()
        self.config["output_dir"] = [output_dir]
        
        # Save config
        config_file = Path(self.config["output_dir"][0], "config.yml")
        with open(config_file, 'w') as file:
            yaml.safe_dump(self.datetime_to_string(self.config), 
                           file, default_style=None, 
                           default_flow_style=False)

        
        print("config was saved as config.yml")


        # Save model_state_dicts to model_state_dict.pt file
        torch.save(self.globalData["model"].model.state_dict(), 
                   Path(self.config["output_dir"][0], "model_state_dict.pt"))
        print("Model state_dict was saved as model_state_dict.pt")
        
        # Save global data
        torch.save(self.globalData, 
                   Path(self.config["output_dir"][0], "globalData.pt"))
        print("globalData was saved as globalData.pt")
        

    def load_project_event(self):
        pass
        
    def datetime_to_string(self, config):
        if "train_period" in config.keys():
            try:
                config["train_period"] = str(config["train_period"].\
                                             strftime('%Y-%m-%d %H:%M'))
                #config["train_period"] = list(config["train_period"])
            except:
                pass
                
        if "test_period" in config.keys():
            try:
                config["test_period"] = str(config["test_period"].\
                                            strftime('%Y-%m-%d %H:%M'))    
                #config["test_period"] = list(config["test_period"])    
            except:
                pass

        if "forecast_period" in config.keys():
            try:
                config["forecast_period"] = str(config["forecast_period"].\
                                                strftime('%Y-%m-%d %H:%M'))
                #config["forecast_period"] = list(config["forecast_period"])
            except:
                pass
                
        if "object_id" in config.keys():
            object_id = []
            for i in config["object_id"]:
                object_id.append(str(i))
                
            config["object_id"] = object_id                
        return config
                