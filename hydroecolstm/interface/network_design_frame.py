
import customtkinter as ctk
from CTkToolTip import CTkToolTip

class NetworkDesignFrame(ctk.CTkFrame):
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
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5)
        self.tabview.grid(row=0, column=0, padx=(20, 20), pady=(10, 10), sticky="ew")
        self.tabview.add("LSTM_DL")
        self.tabview.tab("LSTM_DL").grid_columnconfigure((0,1), weight=1)
        self.tabview.tab("LSTM_DL").rowconfigure((0,1,2,3,4,5,6,7), weight=1)
        #self.tabview.add("RNN")
        #self.tabview.tab("RNN").grid_columnconfigure((0,1), weight=1)
                   
        # ---------------------------------------------content of load data tab
        self.intro_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                                   text="LSTM_DL = Long Short-Term Memory layer + " +
                                                   "Dense (Linear) Layer",
                                                   font=ctk.CTkFont(weight="bold"))
        self.intro_label.pack(anchor="c", pady = (10,10))
        
        
        self.hidden_size_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                                   text="1. Number of hidden units of the LSTM layer")
        self.hidden_size_label.pack(anchor="w", pady = (4,4))
        self.hidden_size = ctk.CTkTextbox(master=self.tabview.tab("LSTM_DL"), 
                                          height=10, width = 140)
        self.hidden_size.bind('<KeyRelease>', self.get_hidden_size)
        self.hidden_size.insert("0.0", "30") 
        self.hidden_size.pack(anchor="w", pady = (4,4))
        
        
        self.nlayers_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                               text="2. Number of LSTM layers")
        self.nlayers_label.pack(anchor="w", pady = (4,4))
        self.nlayers= ctk.CTkOptionMenu(self.tabview.tab("LSTM_DL"),
                                             values=list(map(str,list(range(1,21,1)))),
                                             command=self.get_num_layers) 
        self.nlayers.pack(anchor="w", pady = (4,4))
        
        self.activation_function_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                               text="3. Activation function of Dense (Linear) Layer:")
        self.activation_function_label.pack(anchor="w", pady = (4,4))
        self.activation_function_option= ctk.CTkOptionMenu(self.tabview.tab("LSTM_DL"),
                                             values=["Identity", "ReLu", "Sigmoid",
                                                     "Tanh", "Softplus"],
                                             command=self.get_activation_function_name) 
        self.activation_function_option.pack(anchor="w", pady = (4,4))

        
    def get_hidden_size(self, dummy):
        # Get number of hidden layers
        get_input_text = self.hidden_size.get("0.0", "end")
        # convert input to integer
        self.config["hidden_size"] = int(get_input_text)
        print(self.config["hidden_size"])
             
    # Get number of lstm layers
    def get_num_layers(self, nlayer: str):
        self.config["num_layers"] = int(nlayer)
        print("num_layers = ", self.config["num_layers"])
        
    def get_activation_function_name(self, act: str):
        self.config["activation_function_name"] = act
        
        print("activation_function_name = ", 
              self.config["activation_function_name"])








