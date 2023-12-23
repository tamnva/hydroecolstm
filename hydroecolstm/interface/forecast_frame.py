
import customtkinter as ctk
#import tkinter as tk
#import numpy as np
from CTkListbox import CTkListbox
#from matplotlib.figure import Figure
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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
        self.file_selection_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="1. Input data files")
        self.file_selection_label.grid(row=0, column=0, padx = 10, pady=(5,5), sticky="w")
        
        # Select dynamic data label
        self.dynamic_file_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Select dynamic data file", 
                                               command=None)
        self.dynamic_file_button.grid(row=1, column=0, padx = 10, pady=(5,5), sticky="we")       
    
        self.static_file_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Select static data file", 
                                               command=None)
        self.static_file_button.grid(row=2, column=0, padx = 10, pady=(5,5), sticky="we")  
        
        self.file_check_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="Internal import")
        self.file_check_label.grid(row=0, column=1, padx = 10, pady=(5,5), sticky="w")

        self.dynamic_file_checkvalue = ctk.IntVar(value=0)         
        self.dynamic_file_checkbox = ctk.CTkCheckBox(self.tabview.tab("1. Inputs"), 
                                               text=" ",
                                               command=None, 
                                               variable=self.dynamic_file_checkvalue)
        self.dynamic_file_checkbox.grid(row=1, column=1, padx = 10, pady=(10,10), sticky="we")

        self.static_file_checkvalue = ctk.IntVar(value=0)         
        self.static_file_checkbox = ctk.CTkCheckBox(self.tabview.tab("1. Inputs"), 
                                               text=" ",
                                               command=None, 
                                               variable=self.static_file_checkvalue)
        self.static_file_checkbox.grid(row=2, column=1, padx = 10, pady=(10,10), sticky="we")

        self.model_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                         text="2. Trained model from")
        self.model_label.grid(row=3, column=0, padx = 10, pady=(40,5), sticky="w")
        self.model_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                               anchor='w', 
                                               text="Load saved model", 
                                               command=None)
        self.model_button.grid(row=4, column=0, padx = 10, pady=(5,5), sticky="we")          
        self.model_check_value = ctk.IntVar(value=0)         
        self.model_check = ctk.CTkCheckBox(self.tabview.tab("1. Inputs"), 
                                               text=" ",
                                               command=None, 
                                               variable=self.model_check_value)
        self.model_check.grid(row=4, column=1, padx = 10, pady=(10,10), sticky="we")

        self.object_id_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                 text="3. Select object_id for forecasting")
        self.object_id_label.grid(row=0, column=3, columnspan=1, padx = 10, pady=(5,5), sticky="w")
        self.object_id_forecast = CTkListbox(master=self.tabview.tab("1. Inputs"), 
                                           multiple_selection=True, border_width=1.5,
                                           text_color="black")
        
        self.object_id_forecast.grid(row=1, column=3, rowspan=4, padx = 10, pady=(10,10), sticky="we")

        self.run_label = ctk.CTkLabel(self.tabview.tab("1. Inputs"), 
                                        text="4. Run forecast")
        self.run_label.grid(row=5, column=3, padx = 10, pady=(5,5), sticky="w")
       
        self.run_button = ctk.CTkButton(self.tabview.tab("1. Inputs"), 
                                              anchor='w', 
                                              text="Run", 
                                              command=None)
        self.run_button.grid(row=6, column=3, padx = 10, pady=(5,5), sticky="w") 
        
        
        #----------------------------------------------------------------------
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
        #self.object_id.bind('<KeyRelease>', self.get_object_id)

        self.target_feature = ctk.CTkTextbox(master=self.select_input_frame, height=30)
        self.target_feature.insert("0.0", "target_feature") 
        self.target_feature.grid(row=1, column=0, sticky="w", padx=(5,5), pady=(5,5))
        #self.target_feature.bind('<KeyRelease>', self.get_target_feature)
        
        self.next_object_id_button = ctk.CTkButton(self.select_input_frame, 
                                                   anchor='w',
                                                   command= None, #self.next_object_id, 
                                                   text="Next object id")
        self.next_object_id_button.grid(row=0, column=1, sticky="w", 
                                        padx=(5,5), pady=(5,5))
        self.next_target_feature_button = ctk.CTkButton(self.select_input_frame, 
                                                        anchor='w',
                                                   command=None, #self.next_target_feature, 
                                                   text="Next target features")
        self.next_target_feature_button.grid(row=1, column=1, sticky="w", 
                                             padx=(5,5), pady=(5,5))
                
        self.update_plot = ctk.CTkButton(self.select_input_frame, anchor='w', 
                                 command=None, text="Update plot")
        self.update_plot.grid(row=3, column=0, columnspan=2, sticky="w", padx=(5,20), pady=(10,10))

    # Get dropout
    def test(self):
        return None