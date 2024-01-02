import tkinter as tk
import customtkinter as ctk
from hydroecolstm.model.lstm_linears import Lstm_Linears
from hydroecolstm.model.ea_lstm import Ea_Lstm_Linears
from hydroecolstm.model.train import Train
from CTkToolTip import CTkToolTip

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
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5,
                                      fg_color = "transparent")
        self.tabview.pack(fill='both',expand=1)
        self.tabview.add("Trainer")
        self.tabview.tab("Trainer").grid_columnconfigure((1,1), weight=1)
        
        # ---------------------------------------------content of load data tab
        self.nepoch_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                         text="1. Number of epochs")
        self.nepoch_label.pack(anchor='w')
        
        self.nepoch = ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="5")
               
        self.nepoch.pack(anchor='w')
        self.nepoch.bind('<KeyRelease>', self.get_nepoch)
        CTkToolTip(self.nepoch, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message='Number of training epochs. Input should be numeric ') 
        
        self.learning_rate_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                                text="2. Learning rate")
        self.learning_rate_label.pack(anchor='w', pady=(4, 4))
        self.learning_rate= ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="0.01")
        self.learning_rate.pack(anchor='w')
        self.learning_rate.bind('<KeyRelease>', self.get_learning_rate)
        
        self.warmup_length_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                         text="3. Warm-up length")
        self.warmup_length_label.pack(anchor='w', pady=(4, 4))
        self.warmup_length= ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="20") 
        CTkToolTip(self.warmup_length, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message='The number of timesteps used for warm-up.' +
                   ' For example, the first n simulated outputs will be skipped'+
                   ' when calculating the objective function.') 
        
        self.warmup_length.pack(anchor='w')
        self.warmup_length.bind('<KeyRelease>', self.get_warmup_length)
        
        self.optim_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                        text="4. Optimization method")
        self.optim_label.pack(anchor='w', pady=(4, 4))      
        self.optim = ctk.CTkOptionMenu(self.tabview.tab("Trainer"),
                                                   values=['Adam'],
                                                   command=self.get_optim_method) 
        self.optim.pack(anchor='w')
        
        self.loss_functionlabel = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                               text="5. Loss function")
        self.loss_functionlabel.pack(anchor='w', pady=(4, 4))      
        self.loss = ctk.CTkOptionMenu(self.tabview.tab("Trainer"),
                                                   values=['Root Mean Square Error (RMSE)',
                                                           "Nash–Sutcliffe Efficiency (1-NSE)",
                                                           #"Kling-Gupta Efficiency (1-KGE)",
                                                           'Mean Absolute Error (MAE)',
                                                           'Mean Squared Error (MSE)'],
                                                   command=self.get_objective_function_name) 
        self.loss.pack(anchor='w')
        
        self.run_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                      text="6. Run train test")
        self.run_label.pack(anchor='w', pady=(4, 4))      
        self.run = ctk.CTkButton(self.tabview.tab("Trainer"), anchor='w', 
                                         command=self.run_train_test,
                                         text="Run")
        self.run.pack(anchor='w')
      
        # Progressbar
        self.progressbar = ctk.CTkProgressBar(master=self.tabview.tab("Trainer"))
        self.progressbar.pack(anchor='w', fill = "both", pady=10)
        self.progressbar.configure(mode="determinate", progress_color="orange")
        self.progressbar.set(0)
        #self.progressbar.step()
        
    # Get number of epochs
    def get_nepoch(self, dummy):
        try:
            self.config["n_epochs"] = int(self.nepoch.get().strip())
            print(f"Number of epochs = {self.config['n_epochs']}")
        except:
            tk.messagebox.showinfo(title="Error", message="Input should be integer")
        
    # Get learning_rate
    def get_learning_rate(self, dummy):
        try:
            self.config["learning_rate"]  = float(self.learning_rate.get().strip())
            print(f"Learning rate = {self.config['learning_rate']}")
        except:
            tk.messagebox.showinfo(title="Error", message="Input should be numeric")

    # Get warm up length
    def get_warmup_length(self, dummy):
        try:
            self.config["warmup_length"] =  int(self.warmup_length.get().strip()) 
            print(f"Warm-up length = {self.config['warmup_length']}")  
        except:
            tk.messagebox.showinfo(title="Error", message="Input should be integer")
     
        
    # Get number of lstm layers
    def get_objective_function_name(self, method: str):
        obj_name = {'Root Mean Square Error (RMSE)': "RMSE",
                    "Nash–Sutcliffe Efficiency (1-NSE)": "1-NSE",
                    #"Kling-Gupta Efficiency (1-KGE)": "KGE",
                    'Mean Absolute Error (MAE)': "MAE",
                    'Mean Squared Error (MSE)': "MSE"} 
                              
        self.config["objective_function_name"] = obj_name[method]
        print(self.config["objective_function_name"])

    # Get number of lstm layers
    def get_optim_method(self, method: str):
        self.config["optim_method"] = method
        print(self.config["optim_method"])
     
    def run_train_test(self):
        # Set progress to zero
        self.progressbar.set(0)

        self.run.configure(fg_color='gray')
        self.run.configure(state="disabled")
        
        tk.messagebox.showinfo(title="Message box", 
                               message="Trainning will start after closing this box")
        
        try:
            if self.config["model_class"] == "LSTM":
                self.globalData["model"] = Lstm_Linears(config=self.config)
            else:
                self.globalData["model"] = Ea_Lstm_Linears(config=self.config)
                
            # Train the model
            self.globalData["Train"] = Train(config=self.config, model=self.globalData["model"])
            
            self.globalData["model"], self.globalData["y_train_scale_predict"] =\
                self.globalData["Train"](x=self.globalData["x_train_scale"],y=self.globalData["y_train_scale"])
            
            # Run forward test the model
            self.globalData["y_test_scale_predict"] = self.globalData["model"](self.globalData["x_test_scale"])
                
            # Inverse transform back to the original scale
            
            tk.messagebox.showinfo(title="Message box",
                                   message="Finished training/testing")
        except:
            None

        self.progressbar.set(1.0)        
        self.run.configure(state="normal")
        self.run.configure(fg_color=['#3a7ebf', '#1f538d'])

    
        
        
        