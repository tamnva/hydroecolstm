import matplotlib
import customtkinter as ctk
import tkinter as tk
import torch
from pandastable import Table
from hydroecolstm.data.read_config import read_config
from hydroecolstm.interface.utility import (ToplevelWindow, 
                                            plot_train_valid_loss,
                                            plot_time_series,
                                            check_size,check_linestyle,
                                            check_alpha,check_line_plot,
                                            check_marker, check_color, 
                                            check_ylim,
                                            combine_simulated)
from CTkToolTip import CTkToolTip

    
class VisualizeFrame(ctk.CTkScrollableFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config=config
        self.globalData=globalData
        self.columnconfigure((0), weight=1)
        self.__create_widgets() 
        
        self.plot_window=None
        
    # create widgets for sidebar frame
    def __create_widgets(self):
        # create tabs

        self.tabview=ctk.CTkTabview(self, width=750, border_width=1.5,
                                      fg_color="transparent")
        self.tabview.grid(row=0, column=0, sticky="we")
        self.tabview.add("1. Data for plot")
        self.tabview.tab("1. Data for plot").grid_columnconfigure((0,1),weight=1)
        self.tabview.add("2. Loss plot")
        self.tabview.tab("2. Loss plot").grid_columnconfigure((0,1,2), weight=1)
        self.tabview.add("3. Time-series plot")
        self.tabview.tab("3. Time-series plot").grid_columnconfigure((0,1,2), weight=1)
        #----------------------------------------------------------------------
        self.data_plot_label=ctk.CTkLabel(self.tabview.tab("1. Data for plot"),
                                         text="1. Select data for plot")
        self.data_plot_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        self.data_selection = ctk.CTkOptionMenu(self.tabview.tab("1. Data for plot"),
                                                values=['from current project',
                                                        'from other project'],
                                                command=self.data_selection_event)
        self.data_selection.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        CTkToolTip(self.data_selection, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, justify="left", 
                   message="Select 'from other project' ONLY IF you want to use "+ 
                   "data from other project")
        
        #----------------------------------------------
        self.loss_label=ctk.CTkLabel(self.tabview.tab("2. Loss plot"),
                                            text="1. Plot training and validation loss")
        self.loss_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=5)
        
        self.loss_plot=ctk.CTkButton(self.tabview.tab("2. Loss plot"), anchor='w', 
                                         command=self.loss_plot_event,
                                         text="Plot (update plot)")
        self.loss_plot.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        CTkToolTip(self.loss_plot, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, justify="left", 
                   message='Click here to create plot or update plot (in a new window)')
        
        # check to load/unload
        self.plot_loss=ctk.IntVar(value=0)        
        self.plot_loss_checkbox=ctk.CTkCheckBox(self.tabview.tab("2. Loss plot"), 
                                               text="Show plot settings",
                                               command=self.show_plot_loss_setting, 
                                               variable=self.plot_loss)
        self.plot_loss_checkbox.grid(row=2, column=0, padx=10, pady=(10, 0), 
                                     sticky="w")
        CTkToolTip(self.plot_loss_checkbox, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, 
                   justify="left", message='Change plot settings here or ' + 
                   'leave it empty for default settings')
        
        self.plot_loss_frame=ctk.CTkFrame(self.tabview.tab("2. Loss plot"),
                                            height=400, fg_color="transparent")
        self.plot_loss_frame.columnconfigure((0,1,2), weight=1)
        
        #-----------------------------------------------------Plot loss setting 
        self.common_settings_loss_plot=ctk.CTkLabel(self.plot_loss_frame,
                                         text="Common settings")
        self.common_settings_loss_plot.grid(row=0, column=0, sticky="w", 
                                            padx=5, pady=5)
        
        # Plot title
        self.title=ctk.CTkEntry(self.plot_loss_frame,
                                  placeholder_text="plot title")
        self.title.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        # X label
        self.xlabel=ctk.CTkEntry(self.plot_loss_frame, 
                                   placeholder_text="xlabel")
        self.xlabel.grid(row=2, column=0, sticky="w", padx=5, pady=5)

        # Y label
        self.ylabel=ctk.CTkEntry(self.plot_loss_frame, 
                                   placeholder_text="ylabel")
        self.ylabel.grid(row=3, column=0, sticky="w", padx=5, pady=5)

        # Legend name of the best model
        self.best_model_legend=ctk.CTkEntry(self.plot_loss_frame , 
                                  placeholder_text="best model name")
        self.best_model_legend.grid(row=4, column=0, sticky="w", 
                                    padx=5, pady=5)
        
        #---------------------------------------------------------Training line
        # training line
        self.training_line=ctk.CTkLabel(self.plot_loss_frame,
                                         text="Training loss line")
        self.training_line.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Color of the training loss line
        self.train_line_color=ctk.CTkEntry(self.plot_loss_frame , 
                                  placeholder_text="color")
        CTkToolTip(self.train_line_color, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e', wraplength=500, justify="left", 
                   message=str(list(matplotlib.colors.cnames.keys())))
        
        self.train_line_color.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Line type of the training loss line
        self.train_line_style=ctk.CTkEntry(self.plot_loss_frame , 
                                  placeholder_text="line stype")
        CTkToolTip(self.train_line_style, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, 
                   justify="left", message='solid, dashed, dashdot, dotted')
        
        self.train_line_style.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # Line type of the training loss line
        self.train_legend=ctk.CTkEntry(self.plot_loss_frame, 
                                  placeholder_text="legend name")
        self.train_legend.grid(row=3, column=1, sticky="w", padx=5, pady=5)
              
        #-------------------------------------------------------Validation line
        self.validation_line=ctk.CTkLabel(self.plot_loss_frame,
                                         text="Validation loss line")
        self.validation_line.grid(row=0, column=2, sticky="w", padx=5, pady=5)
        
        # Color of the validation loss line
        self.valid_line_color=ctk.CTkEntry(self.plot_loss_frame , 
                                  placeholder_text="color")
        self.valid_line_color.grid(row=1, column=2, sticky="w", padx=5, pady=5)
        CTkToolTip(self.valid_line_color, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e', wraplength=500, justify="left", 
                   message=str(list(matplotlib.colors.cnames.keys())))
        
        # Line type of the validation loss line
        self.valid_line_style=ctk.CTkEntry(self.plot_loss_frame , 
                                  placeholder_text="line style")
        self.valid_line_style.grid(row=2, column=2, sticky="w", padx=5, pady=5)
        CTkToolTip(self.valid_line_style, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, 
                   justify="left", message='solid, dashed, dashdot, dotted')

        # Line type of the training loss line
        self.valid_legend=ctk.CTkEntry(self.plot_loss_frame , 
                                  placeholder_text="legend name")
        CTkToolTip(self.valid_legend, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, justify="left", 
                   message='Validation line legend: type name of the validation line here, \n' +
                   'or leave it empty to use default name')
        self.valid_legend.grid(row=3, column=2, sticky="w", padx=5, pady=5)
        
        # ---------------------------------------------content of load data tab
        self.object_id_label =\
            ctk.CTkLabel(self.tabview.tab("3. Time-series plot"),
                         text="1. Select object id and target feature for plot")
            
        self.object_id_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=5)

        self.object_id=ctk.CTkTextbox(self.tabview.tab("3. Time-series plot"), 
                                        height=30, border_width=1.5)
        
        self.object_id.insert("0.0", "object_id") 
        self.object_id.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.object_id.bind('<KeyRelease>', self.get_object_id)

        self.target_feature=ctk.CTkTextbox(self.tabview.tab("3. Time-series plot"), 
                                             height=30, border_width=1.5)
        
        self.target_feature.insert("0.0", "target_feature") 
        
        self.target_feature.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        
        self.target_feature.bind('<KeyRelease>', self.get_target_feature)
        
        self.next_object_id_button=ctk.CTkButton(self.tabview.tab("3. Time-series plot"), 
                                                   anchor='w',
                                                   command=self.next_object_id, 
                                                   text="Next object id")
        
        self.next_object_id_button.grid(row=1, column=1, sticky="w", 
                                        padx=5, pady=5)
        
        self.next_target_feature_button =\
            ctk.CTkButton(self.tabview.tab("3. Time-series plot"), anchor='w',
                          command=self.next_target_feature,
                          text="Next target features")
            
        self.next_target_feature_button.grid(row=2, column=1, sticky="w", 
                                             padx=5, pady=5)
        
        # Plot button
        self.update_plot=ctk.CTkButton(self.tabview.tab("3. Time-series plot"), anchor='we', 
                                 command=self.plot_figure, text="Plot (update plot)")
        
        self.update_plot.grid(row=1, column=2,  
                              sticky="we", padx=5, pady=5)
        
        # show all data
        self.show_all_data=ctk.CTkButton(self.tabview.tab("3. Time-series plot"), anchor='w', 
                                         command=self.show_all_data_event,
                                         text="Show data source")
        self.show_all_data.grid(row=2, column=2, sticky="we", padx=5, pady=5)
        CTkToolTip(self.show_all_data, delay=0.1, bg_color='orange',
                   text_color='black', anchor='e',  wraplength=500, justify="left", 
                   message='Click here to show all simulated and observed data')
        
        
        #-------------------------------------------------------2. Plot setting
        # check to load/unload
        self.plot_timeseries=ctk.IntVar(value=0)        
        self.plot_timeseries_checkbox=ctk.CTkCheckBox(self.tabview.tab("3. Time-series plot"), 
                                               text="Show plot settings",
                                               command=self.show_plot_timeseries_setting, 
                                               variable=self.plot_timeseries)
        
        self.plot_timeseries_checkbox.grid(row=3, column=0, padx=10, pady=(10, 0), sticky="w")
        
        self.plot_timeseries_frame=ctk.CTkFrame(self.tabview.tab("3. Time-series plot"),
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
            dataframe = combine_simulated(self.globalData, 
                                          self.globalData["y_column_name"])
            
            self.table = Table(tk.Toplevel(self), dataframe=dataframe, 
                               showtoolbar=True, showstatusbar=True)
            self.table.show()
        except:
            pass
        
        
    # Data selection event
    def data_selection_event(self, method: str):
        if method == "from other project":
            #  Load config file
            self.load_data_label=ctk.CTkLabel(self.tabview.tab("1. Data for plot"),
                                             text="2. Load data from other projects (ATTENTION!!!)")
            self.load_data_label.grid(row=2, column=0, padx=5, pady=5, 
                                      sticky="w")
            self.load_config=ctk.CTkButton(self.tabview.tab("1. Data for plot"),
                                           text="Select config.yml file type",
                                           command=self.load_config_event)
            self.load_config.grid(row=3, column=0, padx=5, pady=5, sticky="w")
            CTkToolTip(self.load_config, delay=0.1, bg_color='orange',
                       text_color='black', anchor='e',  wraplength=500, justify="left", 
                       message='Click this button to load config.yml from other '+
                       'project and OVERWRITE configuration created from previous steps ' +
                       '(if they exist)')
            
            self.selected_config_file = ctk.CTkLabel(self.tabview.tab("1. Data for plot"),
                                                          text="No file was selected")
            self.selected_config_file.grid(row=3, column=1, padx=5, pady=5, sticky="w")             
 
    
            self.load_data=ctk.CTkButton(self.tabview.tab("1. Data for plot"),
                                           text="Select data.pt file type",
                                           command=self.load_data_event)
            self.load_data.grid(row=4, column=0, padx=5, pady=5, sticky="w") 
            CTkToolTip(self.load_config, delay=0.1, bg_color='orange',
                       text_color='black', anchor='e',  wraplength=500, justify="left", 
                       message='Click this button to load data.pt from other '+
                       'project and OVERWRITE data created from previous steps ' +
                       '(if they exist)')
            self.selected_data_file = ctk.CTkLabel(self.tabview.tab("1. Data for plot"),
                                                          text="No file was selected")
            self.selected_data_file.grid(row=4, column=1, padx=5, pady=5, sticky="w") 
        else:
            # hide all buttons for load data from previous project
            self.load_data_label.grid_forget()
            self.load_config.grid_forget()
            self.selected_config_file.grid_forget()
            self.load_data.grid_forget()
            self.selected_data_file.grid_forget()
            
    
    def load_config_event(self):
        try:
            file_name=ctk.filedialog.askopenfilename(title="Select config.yml file type", 
                                                     filetypes=(('yml files', '*.yml'),
                                                                ('All files', '*.*')))
            self.config.update(read_config(file_name))
            self.selected_config_file.configure(text= '...' + file_name[-30:])
        except:
            pass
        
    def load_data_event(self):
        try:
            file_name=ctk.filedialog.askopenfilename(title="Select data.pt file type", 
                                                     filetypes=(('pt files', '*.pt'),
                                                                ('All files', '*.*')))
            self.globalData.update(torch.load(file_name))
            self.selected_data_file.configure(text= '...' + file_name[-30:])
            print(self.globalData.keys())
        except:
            pass
        
        
    # Show plot loss setting
    def show_plot_loss_setting(self):
        if self.plot_loss.get() == 0:
            self.plot_loss_frame.grid_forget()
        else:
            self.plot_loss_frame.grid(row=3, column=0, columnspan=3,
                                      sticky="w", padx=(0,0), pady=(20,20))

    # Show plot time series setting
    def show_plot_timeseries_setting(self):
        if self.plot_timeseries.get() == 0:
            self.plot_timeseries_frame.grid_forget()
        else:
            self.plot_timeseries_frame.grid(row=4, column=0, columnspan=3,
                                      sticky="w", padx=(0,0), pady=(20,20))
            
    # Get next object id
    def next_object_id(self):
        try:
            if self.globalData["object_id_no"] > len(self.config["object_id"]) - 1:
                self.globalData["object_id_no"]=0
            
            self.object_id.delete("0.0", "end")
            
            self.object_id.insert("0.0", self.config["object_id"]
                                  [self.globalData["object_id_no"]])
            
            self.globalData["object_id_plot"] =\
                str(self.config["object_id"][self.globalData["object_id_no"]])
                
            self.globalData["object_id_no"] += 1
        except:
            pass

    def next_target_feature(self):
        try:
            if self.globalData["target_feature_no"] > len(self.config["target_features"]) - 1:
                self.globalData["target_feature_no"]=0
               
            self.target_feature.delete("0.0", "end")
            self.target_feature.insert("0.0", self.config["target_features"]
                                       [self.globalData["target_feature_no"]])
            self.globalData["target_feature_plot"] =\
                str(self.config["target_features"][self.globalData["target_feature_no"]])
                
            self.globalData["target_feature_no"] += 1
        except:
            pass
            
    def get_object_id(self, dummy):
        self.globalData["object_id_plot"]=str(self.object_id.get("0.0", "end"))
        self.globalData["object_id_plot"]=self.globalData["object_id_plot"].strip()
        print(f"Selected object_id for plot={self.globalData['object_id_plot']}")

    def get_target_feature(self, dummy):
        self.globalData["target_feature_plot"] =\
            str(self.target_feature.get("0.0", "end"))
            
        self.globalData["target_feature_plot"] =\
            self.globalData["target_feature_plot"].strip()
            
        print(f"Selected target_feature for plot={self.globalData['target_feature_plot']}")
       
    def plot_figure(self):
        try:
            # delete later
            key=self.globalData["object_id_plot"]
            idx=self.config["target_features"].index(self.globalData["target_feature_plot"])
            
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
            if ylabel == "": ylabel=self.globalData["target_feature_plot"]
            ylim_up=check_ylim(self.ylim_up, -9999.0)
                
            plot_window=ToplevelWindow(window_name="Plot window")
            
            # Plot
            plot_time_series(plot_window, self.globalData, key, idx, lineplot, 
                             color_obs, alpha_obs, size_obs, linestyle_obs,
                             marker_obs, label_obs, color_sim, alpha_sim, 
                             size_sim, linestyle_sim, label_sim, title, xlabel,
                             ylabel, ylim_up, forecast_period=False)
            
            
        except:
            tk.messagebox.showinfo(title="Error", 
                                   message="Cannot plot time series")
        
    def loss_plot_event(self):
        try:
            # Data for plot

            loss=self.globalData["trainer"].loss

            # Make plot window
            plot_window=ToplevelWindow(window_name="Plot window")
            
            # Now plot
            plot_train_valid_loss(loss, 
                                  plot_window, 
                                  self.xlabel.get(),
                                  self.ylabel.get(), 
                                  self.title.get(),
                                  self.train_line_color.get().strip(),
                                  self.train_line_style.get().strip(),
                                  self.train_legend.get().strip(),
                                  self.valid_line_color.get().strip(),
                                  self.valid_line_style.get().strip(),
                                  self.valid_legend.get().strip(),
                                  self.best_model_legend.get().strip())
        except:
            # Make plot window
            #plot_window=ToplevelWindow(window_name="Plot window")
            #plot_time_series(plot_window)
            
            tk.messagebox.showinfo(title="Error", 
                                   message="Cannot plot loss")
            
    def show_loss_pandastable(self):
        try:
            self.table = Table(tk.Toplevel(self), dataframe=self.globalData['dynamic_data'], 
                               showtoolbar=True, showstatusbar=True)
            self.table.show()
        except:
            None

