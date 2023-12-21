import tkinter as tk
import customtkinter as ctk
from hydroecolstm.model.lstms import LSTM_DL

class TrainTestFrame(ctk.CTkScrollableFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config = config
        self.globalData = globalData
        self.columnconfigure(0, weight=1)
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self): 
        
        # create tabs
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5)
        self.tabview.grid(row=0, column=0, padx=(20, 20), pady=(10, 10), sticky="ew")
        self.tabview.add("LSTM_DL")
        self.tabview.tab("LSTM_DL").grid_columnconfigure((1,1), weight=1)
        #self.tabview.add("RNN")
        #self.tabview.tab("RNN").grid_columnconfigure((0,1), weight=1)
        
        # ---------------------------------------------content of load data tab
        self.nepoch_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                         text="1. Number of epochs")
        self.nepoch_label.pack(anchor='w')
        self.nepoch = ctk.CTkTextbox(master=self.tabview.tab("LSTM_DL"), 
                                               height=30)
        self.nepoch.insert("0.0", "50") 
        self.nepoch.pack(anchor='w')
        self.nepoch.bind('<KeyRelease>', self.get_nepoch)
        
        self.learning_rate_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                                text="2. Learning rate")
        self.learning_rate_label.pack(anchor='w', pady=(4, 4))
        self.learning_rate= ctk.CTkTextbox(master=self.tabview.tab("LSTM_DL"),
                                          height=30)
        self.learning_rate.insert("0.0", "0.01") 
        self.learning_rate.pack(anchor='w')
        self.learning_rate.bind('<KeyRelease>', self.get_learning_rate)        
        
        self.dropout_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                          text="3. Dropout")
        self.dropout_label.pack(anchor='w', pady=(4, 4))
        self.dropout= ctk.CTkTextbox(master=self.tabview.tab("LSTM_DL"),
                                          height=30)
        self.dropout.insert("0.0", "0.30") 
        self.dropout.pack(anchor='w')
        self.dropout.bind('<KeyRelease>', self.get_dropout)

        self.warmup_length_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                         text="4. Warm-up length")
        self.warmup_length_label.pack(anchor='w', pady=(4, 4))
        self.warmup_length= ctk.CTkTextbox(master=self.tabview.tab("LSTM_DL"),
                                          height=30)
        self.warmup_length.insert("0.0", "20") 
        self.warmup_length.pack(anchor='w')
        self.warmup_length.bind('<KeyRelease>', self.get_warmup_length)
        
        self.optim_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                        text="5. Optimization method")
        self.optim_label.pack(anchor='w', pady=(4, 4))      
        self.optim = ctk.CTkOptionMenu(self.tabview.tab("LSTM_DL"),
                                                   values=['Adam'],
                                                   command=self.get_optim_method) 
        self.optim.pack(anchor='w')
        
        self.loss_functionlabel = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                               text="5. Optimization method")
        self.loss_functionlabel.pack(anchor='w', pady=(4, 4))      
        self.loss = ctk.CTkOptionMenu(self.tabview.tab("LSTM_DL"),
                                                   values=['Root Mean Square Error (RMSE)',
                                                           "Nash–Sutcliffe Efficiency (1-NSE)",
                                                           #"Kling-Gupta Efficiency (1-KGE)",
                                                           'Mean Absolute Error (MAE)',
                                                           'Mean Squared Error (MSE)'],
                                                   command=self.get_objective_function_name) 
        self.loss.pack(anchor='w')
        
        self.run_label = ctk.CTkLabel(self.tabview.tab("LSTM_DL"), 
                                      text="6. Run train test")
        self.run_label.pack(anchor='w', pady=(4, 4))      
        self.run = ctk.CTkButton(self.tabview.tab("LSTM_DL"), anchor='w', 
                                         command=self.run_train_test,
                                         text="Run")
        self.run.pack(anchor='w')
      
        #self.slider_progressbar_frame = ctk.CTkFrame(self, fg_color="transparent",height=10)
        #self.slider_progressbar_frame.pack(fill='both',expand=1)
        #self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        #self.progressbar = ctk.CTkProgressBar(master=self.tabview.tab("LSTM_DL"))
        #self.progressbar.pack(fill='both',expand=1)
        #self.progressbar.configure(mode="determinate",progress_color="orange")
        #self.progressbar.set(0)
        #self.progressbar.step()
        
    # Get number of epochs
    def get_nepoch(self, dummy):
        get_n_epochs = self.nepoch.get("0.0", "end")
        self.config["n_epochs"] = int(get_n_epochs)
        print(f"Number of epochs = {self.config['n_epochs']}")
        
    # Get learning_rate
    def get_learning_rate(self, dummy):
        get_learning_rate = self.learning_rate.get("0.0", "end")
        self.config["learning_rate"] = float(get_learning_rate)
        print(f"Learning rate = {self.config['learning_rate']}")

    # Get dropout
    def get_dropout(self, dummy):
        get_dropout = self.dropout.get("0.0", "end")
        self.config["dropout"] = float(get_dropout)
        print(f"Dropout = {self.config['dropout']}")

    # Get warm up length
    def get_warmup_length(self, dummy):
        get_warmup_length = self.warmup_length.get("0.0", "end")
        self.config["warmup_length"] = int(get_warmup_length)
        print(f"Warm-up length = {self.config['warmup_length']}")       
        
    # Get number of lstm layers
    def get_objective_function_name(self, method: str):
        obj_name = {'Root Mean Square Error (RMSE)': "RMSE",
                    "Nash–Sutcliffe Efficiency (1-NSE)": "NSE",
                    #"Kling-Gupta Efficiency (1-KGE)": "KGE",
                    'Mean Absolute Error (MAE)': "MAE",
                    'Mean Squared Error (MSE)': "MSE"} 
        
        print(method)                       
        self.config["objective_function_name"] = obj_name[method]
        print(self.config["objective_function_name"])

    # Get number of lstm layers
    def get_optim_method(self, method: str):
        self.config["optim_method"] = method
        print(self.config["optim_method"])
     
    def run_train_test(self):
        #
        self.run.configure(fg_color='gray')
        self.run.configure(state="disabled")
        
        tk.messagebox.showinfo(title="Message box", 
                               message="Trainning will start after closing this box")
        
        #self.progressbar.set(0.1)
        self.globalData["model"] = LSTM_DL(config=self.config)
        #self.progressbar.set(0.2)
        
        # Train the model
        _, self.globalData["y_train_scale_predict"] =\
            self.globalData["model"].train(x_train=self.globalData["x_train_scale"],
                                           y_train=self.globalData["y_train_scale"])
        #self.progressbar.set(0.9)
        
        # Run forward test the model
        self.globalData["y_test_scale_predict"] =\
            self.globalData["model"].forward(self.globalData["x_test_scale"])

        #self.progressbar.set(1.0)        
        self.run.configure(state="normal")
        self.run.configure(fg_color=['#3a7ebf', '#1f538d'])
        
        tk.messagebox.showinfo(title="Message box", 
                               message="Finished training/testing")
        
        
        
        
        
        
        
        
        
        
        