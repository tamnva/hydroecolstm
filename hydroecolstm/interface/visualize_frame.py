
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
        self.columnconfigure((0,1), weight=1)
        self.rowconfigure((0,1), weight=1)
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self): 
        # ---------------------------------------------content of load data tab
        self.right_frame = ctk.CTkCanvas(master=self, width=400, height=400)
        self.right_frame.grid(row=0, column=1, sticky="e", padx=(20,20), pady=(20,20))        

        self.left_frame = ctk.CTkFrame(master=self, width=400, height=400)
        self.left_frame.grid(row=0, column=0, sticky="w", padx=(20,20), pady=(20,20))

        self.object_id = ctk.CTkTextbox(master=self.left_frame, height=30)
        self.object_id.insert("0.0", "object_id") 
        self.object_id.grid(row=0, column=0, sticky="w", padx=(20,20), pady=(20,20))

        self.show_plot = ctk.CTkButton(self.left_frame, anchor='w', 
                                 command=self.plot_figure, text="Plot")
        self.show_plot.grid(row=1, column=0, sticky="w", padx=(20,20), pady=(20,20))
        
        '''
        x = 4 + np.random.normal(0, 2, 24)
        y = 4 + np.random.normal(0, 2, len(x))
        sizes = np.random.uniform(15, 80, len(x))
        colors = np.random.uniform(15, 80, len(x))
        
        # create a figure
        figure = Figure(figsize=(6, 4), dpi=100)
        figure_canvas = FigureCanvasTkAgg(figure, self.right_frame )
        NavigationToolbar2Tk(figure_canvas, self.right_frame )
        
        axes = figure.add_subplot()
        
        axes.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
        axes.set(xlim=(0, 8), xticks=np.arange(1, 8),
               ylim=(0, 8), yticks=np.arange(1, 8))
        
        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        '''
        
        
    def display_static_data(self):
       data = pd.read_csv(self.config['static_data_file'], delimiter=",", header=0)
       data = data.describe()
       self.table = Table(self.viz_frame, dataframe=data, showtoolbar=True, showstatusbar=True)
       self.table.show()  
       
    def plot_figure(self):
        x = 4 + np.random.normal(0, 2, 24)
        y = 4 + np.random.normal(0, 2, len(x))
        sizes = np.random.uniform(15, 80, len(x))
        colors = np.random.uniform(15, 80, len(x))
        
        # create a figure
        figure = Figure(figsize=(6, 4), dpi=100)
        figure_canvas = FigureCanvasTkAgg(figure, self.right_frame )
        NavigationToolbar2Tk(figure_canvas, self.right_frame )
        
        axes = figure.add_subplot()
        
        axes.scatter(x, y, s=sizes, c=colors, vmin=0, vmax=100)
        axes.set(xlim=(0, 8), xticks=np.arange(1, 8),
               ylim=(0, 8), yticks=np.arange(1, 8))
        
        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        