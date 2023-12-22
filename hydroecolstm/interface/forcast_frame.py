
import customtkinter as ctk
import tkinter as tk
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class ForcastFrame(ctk.CTkFrame):
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
        self.object_id_label = ctk.CTkLabel(self, 
                                            text="Underconstruction: Running the" +
                                            " trained model for prediction or for ungauged basins")
        self.object_id_label.grid(row=0, column=0, sticky="w", padx=(5,5))
        
    
    # Get dropout
    def test(self):
        return None