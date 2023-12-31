
import customtkinter as ctk
import tkinter as tk
from CTkToolTip import CTkToolTip

class NetworkDesignFrame(ctk.CTkScrollableFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config = config
        self.globalData = globalData
        self.columnconfigure((0), weight=1)
        #self.rowconfigure((0,1,2,3), weight=0)
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self): 
        
        # create tabs
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5,
                                      fg_color = "transparent")
        self.tabview.pack(fill='both',expand=1)
        self.tabview.add("Model class")
        self.tabview.tab("Model class").grid_columnconfigure((0,1), weight=1)
        self.tabview.tab("Model class").rowconfigure((0,1,2,3,4,5,6,7), weight=1)
        self.tabview.add("Model head")
        self.tabview.tab("Model head").grid_columnconfigure((0), weight=1)
                   
        # ---------------------------------------------content of load data tab
        self.intro_label = ctk.CTkLabel(self.tabview.tab("Model class"), 
                                                   text="1. Select model class type")
        self.intro_label.pack(anchor="w", pady = (10,10))
        
        self.model_class_type =\
            ctk.CTkOptionMenu(self.tabview.tab("Model class"),
                              values=["LSTM","EA-LSTM"], command=self.get_model_class)
                
        self.model_class_type.pack(anchor="w",  pady = 5)
                
        self.hidden_size_label = ctk.CTkLabel(self.tabview.tab("Model class"), 
                                                   text="2. Number of hidden units of the LSTM layer")
        self.hidden_size_label.pack(anchor="w", pady = (4,4))
        
        self.hidden_size = ctk.CTkEntry(master=self.tabview.tab("Model class"),
                                        placeholder_text="30")
        self.hidden_size.pack(anchor="w", pady = (4,4))
        self.hidden_size.bind('<KeyRelease>', self.get_hidden_size)

        self.nlayers_label = ctk.CTkLabel(self.tabview.tab("Model class"), 
                                               text="3. Number of LSTM layers")
        self.nlayers_label.pack(anchor="w", pady = (4,4))
        self.nlayers= ctk.CTkOptionMenu(self.tabview.tab("Model class"),
                                             values=list(map(str,list(range(1,21,1)))),
                                             command=self.get_num_layers) 
        self.nlayers.pack(anchor="w", pady = (4,4))
        
        self.dropout_label = ctk.CTkLabel(self.tabview.tab("Model class"), 
                                               text="4. Drop out rate")
        self.dropout_label.pack(anchor="w", pady = (4,4))
        self.dropout = ctk.CTkEntry(master=self.tabview.tab("Model class"),
                                        placeholder_text="0.30")
        self.dropout.pack(anchor="w", pady = (4,4))
        self.dropout.bind('<KeyRelease>', self.get_dropout)


        # ----------------------------------------------------------Model heads
        self.intro_label = ctk.CTkLabel(self.tabview.tab("Model head"), 
                                                   text="1. Select model head")
        self.intro_label.grid(row=0, column=0, padx = 10, pady=(5,5), sticky="w")
        
        self.model_head_type =\
            ctk.CTkOptionMenu(self.tabview.tab("Model head"),
                              values=["Regression (REG)",
                                      "Gaussian Mixture Model (GMM)"],
                              command=self.get_model_head_name)
        
        self.model_head_type.grid(row=1, column=0, padx = 10, pady=5, sticky="w")        

        # --------------------------------------Frame for regression model head
        self.regression_frame = ctk.CTkFrame(master=self.tabview.tab("Model head"), 
                                             fg_color = "transparent", border_width=0.0)
        self.regression_frame.grid_columnconfigure((0,1), weight=1)
        self.regression_frame.grid(row=2, column=0, sticky="w", pady=(5, 5))
                
        # Number of layers
        self.regression_nlayers_label = ctk.CTkLabel(self.regression_frame,
                                  text="2. Number of layers")
        self.regression_nlayers_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        self.regression_nlayers = ctk.CTkOptionMenu(self.regression_frame,
                          values=list(map(str, range(1, 6))),
                          command=self.create_entry_regression_model)
        self.regression_nlayers.grid(row=1, column=0, sticky="w", padx=10, pady=5)        
        
        # Label
        
        self.regression_config_label1 = ctk.CTkLabel(self.regression_frame,
                                  text="3. Number of neurons layer ith")
        self.regression_config_label1.grid(row=2, column=0, sticky="w", padx=10, pady=5)      
        self.regression_config_label2 = ctk.CTkLabel(self.regression_frame,
                                  text="4. Activation function layer ith")
        self.regression_config_label2.grid(row=2, column=1, sticky="w", padx=10, pady=5)

        self.option_menu = ctk.CTkOptionMenu(self.regression_frame,
                                             values=["Identity", "ReLu", "Sigmoid",
                                                     "Tanh", "Softplus"],
                                             command=self.reg_get_acts_1)
        self.option_menu.grid(row=3, column=1, sticky="e", padx=10, pady=5)
        self.reg_activation_func = [self.option_menu]
                
        # --------------------------------------Frame for GMM
        self.gmm_frame = ctk.CTkFrame(master=self.tabview.tab("Model head"), 
                                             fg_color = "transparent", border_width=0.0)
        self.gmm_frame.grid_columnconfigure((0,1), weight=1)
        #self.gmm_frame.grid(row=2, column=0, sticky="w", pady=(5, 5))
                
        # Number of neuron in hidden layer
        self.gmm_hidden_size_label = ctk.CTkLabel(self.gmm_frame,
                                  text="2. Work in progress (comming soon)...")
        self.gmm_hidden_size_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.gmm_hidden_size = ctk.CTkEntry(master=self.gmm_frame,
                                            placeholder_text="None")
        self.gmm_hidden_size.grid(row=1, column=0, sticky="w", padx=10, pady=5)        
        # ---------------------------------------------content of load data tab        
    def get_hidden_size(self, dummy):
        # Get number of hidden layers
        try:
            get_input_text = int(self.hidden_size.get().strip())
            self.config["hidden_size"] = get_input_text
            print("hidden size = ", self.config["hidden_size"])
        except:
            pass
             
    # Get number of lstm layers
    def get_num_layers(self, nlayer: str):
        try:
            self.config["num_layers"] = int(nlayer)
            print("num_layers = ", self.config["num_layers"])
        except:
            pass

    # Get number of lstm layers
    def get_dropout(self, dummy: str):
        try:
            self.config["dropout"] = float(self.dropout.get().strip())
            print("Dropout rate = ", self.config["dropout"])
        except:
            pass

    def get_model_head_name(self, model_head_name: str):
        model_head_names = {"Regression (REG)" : "REG",
                            "Gaussian Mixture Model (GMM)": "GMM"}
        
        self.globalData["model_head"] = model_head_names[model_head_name]
        print("Model head name = ", self.globalData["model_head"])
            
        # Delete config if other model head option is 
        if self.globalData["model_head"] == "REG":
                       
            self.regression_frame.grid(row=2, column=0, sticky="w", pady=(5, 5))
            self.gmm_frame.grid_forget()
            
            self.config["REG"] = {}
        else:
            # Hide regression frame
            self.regression_frame.grid_forget()
            self.gmm_frame.grid(row=2, column=0, sticky="w", pady=(5, 5))
            try:
                del self.config["REG"]
            except:
                pass
 
    def get_model_class(self, model_class: str):
        self.config["model_class"] = model_class
        print("Model class name = ", self.config["model_class"])
        
        if model_class == "EA-LSTM":
            tk.messagebox.showinfo(title="Message box", 
                                   message="This model class takes a lot of time to train" +
                                   " need to improve the forward pass for this model class")         
            
    def create_entry_regression_model(self, nlayers: str):
        print("number of layer", int(nlayers))
        
        # try to hide all entry
        try:
            for entry in self.reg_neurons:
                entry.grid_forget()
        except:
            pass
        
        # try to hide all entry
        try:
            for entry in self.reg_activation_func:
                entry.grid_forget()
        except:
            pass
        
        # Now add entry according to the number of layers
        if "REG" in self.config.keys():
            
            row_number = 3
            self.config["REG"]["num_layers"] = int(nlayers)
            self.reg_neurons = []
            self.reg_activation_func = []
            self.config["REG"]["num_neurons"] = [None for i in range(int(nlayers))]
            self.config["REG"]["activation_function"] = ["Identity" for i in 
                                                         range(int(nlayers))]

            get_neurons = {0: self.reg_get_neurons_1, 1: self.reg_get_neurons_2,
                           2: self.reg_get_neurons_3, 3: self.reg_get_neurons_4}
            get_acts = {0: self.reg_get_acts_1, 1: self.reg_get_acts_2,
                        2: self.reg_get_acts_3, 3: self.reg_get_acts_4,
                        4: self.reg_get_acts_5}
            
            for i in range(int(nlayers)):

                if i < int(nlayers) - 1:
                    entry = ctk.CTkEntry(master=self.regression_frame,
                                         placeholder_text="10")
                    entry.grid(row=row_number, column=0, sticky="e", padx=10, pady=5)
                    self.reg_neurons.append(entry)
                    self.reg_neurons[i].bind('<KeyRelease>', get_neurons[i])
                
                option_menu = ctk.CTkOptionMenu(self.regression_frame,
                                                     values=["Identity", "ReLu", "Sigmoid",
                                                             "Tanh", "Softplus"],
                                                     command=get_acts[i])
                option_menu.grid(row=row_number, column=1, sticky="e", padx=10, pady=5)
                self.reg_activation_func.append(option_menu)
                row_number += 1
    
    # Function to get number of input neurons each layer - Make this code clearner
    def reg_get_neurons_1(self, dummy):
        try: 
            self.config["REG"]["num_neurons"][0] = int(self.reg_neurons[0].get().strip())
            print("Num of neurons layer 1 = ",  self.config["REG"]["num_neurons"][0])
        except:
            tk.messagebox.showinfo(title="Error", 
                                   message="Invalid input number of neuron for layer 1")
    def reg_get_neurons_2(self, dummy):
        try: 
            self.config["REG"]["num_neurons"][1] = int(self.reg_neurons[1].get().strip())
            print("Num of neurons layer 2 = ", self.config["REG"]["num_neurons"][1])
        except:
            tk.messagebox.showinfo(title="Error", 
                                   message="Invalid input number of neuron for layer 2")
    def reg_get_neurons_3(self, dummy):
        try: 
            self.config["REG"]["num_neurons"][2] = int(self.reg_neurons[2].get().strip())
            print("Num of neurons layer 3 = ", self.config["REG"]["num_neurons"][2])
        except:
            tk.messagebox.showinfo(title="Error", 
                                   message="Invalid input number of neuron for layer 3")
    def reg_get_neurons_4(self, dummy):
        try: 
            self.config["REG"]["num_neurons"][3] = int(self.reg_neurons[3].get().strip())
            print("Num of neurons layer 4 = ",  self.config["REG"]["num_neurons"][3])
        except:
            tk.messagebox.showinfo(title="Error", 
                                   message="Invalid input number of neuron for layer 4")
    
    # Function to get activation function of each layer - Make this code clearner
    def reg_get_acts_1(self, dummy):
        self.config["REG"]["activation_function"][0] = dummy
        print("Activation function of layer 1 = ", dummy)
    def reg_get_acts_2(self, dummy):
        self.config["REG"]["activation_function"][1] = dummy
        print("Activation function of layer 2 = ", dummy)
    def reg_get_acts_3(self, dummy):
        self.config["REG"]["activation_function"][2] = dummy
        print("Activation function of layer 3 = ", dummy)
    def reg_get_acts_4(self, dummy):
        self.config["REG"]["activation_function"][3] = dummy
        print("Activation function of layer 4 = ", dummy)
    def reg_get_acts_5(self, dummy):
        self.config["REG"]["activation_function"][5] = dummy
        print("Activation function of layer 5 = ", dummy)



