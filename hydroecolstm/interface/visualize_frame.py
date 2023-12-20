
import customtkinter as ctk
import pandas as pd
from pandastable import Table
import tkinter as tk
import numpy as np


from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk



class VisualizeFrame(ctk.CTkFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config = config
        self.globalData = globalData
        self.columnconfigure((0), weight=1)
        self.rowconfigure((0,1), weight=1)
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self): 
        # ---------------------------------------------content of load data tab
        self.object_id_label = ctk.CTkLabel(self, text="1. Please insert object_id for plot")
        self.object_id_label.pack(pady=(5,5), anchor='w')
        self.left_frame = ctk.CTkFrame(master=self, height=400)
        self.left_frame.pack(pady=(5,20), anchor = "w") #grid(row=0, column=0, sticky="w", padx=(20,20), pady=(20,20))

        self.object_id_label = ctk.CTkLabel(self, text="2. Plotting area")
        self.object_id_label.pack(pady=(0,5), anchor='w')
        self.right_frame = ctk.CTkCanvas(master=self, height=400)
        self.right_frame.pack(pady=(5,20), anchor = "w") #grid(row=1, column=0, sticky="w", padx=(20,20), pady=(20,20))  


        self.object_id = ctk.CTkTextbox(master=self.left_frame, height=30)
        self.object_id.insert("0.0", "object_id") 
        self.object_id.pack(pady=(5,5), padx=(5,5), anchor = "w") #grid(row=0, column=0, sticky="w", padx=(20,20), pady=(20,20))
        self.object_id.bind('<KeyRelease>', self.get_object_id)

        self.target_feature = ctk.CTkTextbox(master=self.left_frame, height=30)
        self.target_feature.insert("0.0", "target_feature") 
        self.target_feature.pack(pady=(5,5), padx=(5,5), anchor = "w") #grid(row=0, column=0, sticky="w", padx=(20,20), pady=(20,20))
        self.target_feature.bind('<KeyRelease>', self.get_target_feature)
        
        self.show_plot = ctk.CTkButton(self.left_frame, anchor='w', 
                                 command=self.plot_figure, text="Update plot")
        self.show_plot.pack(pady=(5,5), padx=(5,5), anchor = "w") #grid(row=1, column=0, sticky="w", padx=(20,20), pady=(20,20))
        
    
    # Get dropout
    def get_object_id(self, dummy):
        self.globalData["object_id_plot"] = str(self.object_id.get("0.0", "end"))
        self.globalData["object_id_plot"] = self.globalData["object_id_plot"].strip()
        print(f"Selected object_id for plot = {self.globalData['object_id_plot']}")

    def get_target_feature(self, dummy):
        self.globalData["target_feature_plot"] = str(self.target_feature.get("0.0", "end"))
        self.globalData["target_feature_plot"] = self.globalData["target_feature_plot"].strip()
        print(f"Selected target_feature for plot = {self.globalData['target_feature_plot']}")
        
    def display_static_data(self):
       data = pd.read_csv(self.config['static_data_file'], delimiter=",", header=0)
       data = data.describe()
       self.table = Table(self.viz_frame, dataframe=data, showtoolbar=True, showstatusbar=True)
       self.table.show()  
       
    def plot_figure(self):
        
        # Remove and create frame again to update figure
        self.right_frame.destroy()
        self.right_frame = ctk.CTkFrame(master=self, height=400)
        self.right_frame.pack(pady=(5,20), anchor = "w") 
        
        try:           
            obs = self.globalData["y_test_scale"][self.globalData["object_id_plot"]]  
            obs = obs[:, self.config["target_features"].\
                      index(self.globalData["target_feature_plot"])]  
                
            predict = self.globalData["y_test_scale_predict"]\
                [self.globalData["object_id_plot"]].detach().numpy()
            predict = predict[:, self.config["target_features"].\
                              index(self.globalData["target_feature_plot"])]    
            
            figure = Figure(figsize=(15, 4), dpi=100)
            figure_canvas = FigureCanvasTkAgg(figure, self.right_frame )
            NavigationToolbar2Tk(figure_canvas, self.right_frame )          
            axes = figure.add_subplot()
            axes.plot(obs, 'ro', label = "Observed (test data)", 
                      alpha=0.9, markersize=2.5 )            
            axes.plot(predict, color = 'blue', label = "Predicted (test data)", 
                      alpha=0.9, linewidth=0.75)
            axes.set_title(f"object_id = {self.globalData['object_id_plot']}")
            axes.legend() 
            figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
        except:
            x = 4 + np.random.normal(0, 2, 24)
            y = 4 + np.random.normal(0, 2, len(x))
            figure = Figure(figsize=(15, 4), dpi=100)
            figure_canvas = FigureCanvasTkAgg(figure, self.right_frame )
            NavigationToolbar2Tk(figure_canvas, self.right_frame )
            axes = figure.add_subplot()            
            axes.plot(x, color = 'blue', label = "Predicted (test data)", 
                      alpha=0.9, linewidth=0.75)
            axes.plot(y, 'ro', label = "Observed (test data)", 
                      alpha=0.9, markersize=2.5 )
            axes.legend()
            axes.set_title("Test plot")
                
            figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)            
