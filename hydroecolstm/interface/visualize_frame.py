
import customtkinter as ctk
import tkinter as tk
import numpy as np
from pathlib import Path
import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .utility import config_to_text, sort_key



class VisualizeFrame(ctk.CTkScrollableFrame):
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
        self.select_input_frame = ctk.CTkFrame(master=self, height=400, fg_color = "transparent")
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
        self.save_project.grid(row=3, column=2, sticky="w", padx=(5,5), pady=(0,0))
        #self.load_project = ctk.CTkButton(self.select_input_frame, anchor='w',
        #                                  command=self.load_project_event, 
        #                                  text="Load project") 
        #self.load_project.grid(row=3, column=3, sticky="e", padx=(5,5), pady=(20,20))
    
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
        out_file = Path(self.config["output_dir"][0], "config.yml")
        self.write_yml_file(config=self.config, out_file=out_file)
        print("config was saved as config.yml")

        # Save model_state_dicts to model_state_dict.pt file
        torch.save(self.globalData["model"].state_dict(), 
                   Path(self.config["output_dir"][0], "model_state_dict.pt"))
        print("Model state_dict was saved as model_state_dict.pt")
        
        # Save global data
        torch.save(self.globalData, 
                   Path(self.config["output_dir"][0], "globalData.pt"))
        print("globalData was saved as globalData.pt")
        
    def write_yml_file(self, config, out_file):
        # Convert config to text
        output_text = config_to_text(config=sort_key(self.config))
        
        # Write config to config file
        with open(out_file, "w") as config_file:
            for line in output_text:
                config_file.write(line)
