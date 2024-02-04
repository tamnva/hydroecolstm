import tkinter as tk
import customtkinter as ctk
from hydroecolstm.model.lstm_linears import Lstm_Linears
from hydroecolstm.model.ea_lstm import Ea_Lstm_Linears
from hydroecolstm.train.trainer import Trainer
from CTkToolTip import CTkToolTip

class TrainTestFrame(ctk.CTkScrollableFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config = config
        self.globalData = globalData
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self): 
        
        # create tabs
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5,
                                      fg_color = "transparent")
        self.tabview.pack(fill='both',expand=1)
        self.tabview.add("Trainer")
        self.tabview.tab("Trainer").grid_columnconfigure((0,1), weight=1)
        
        # ---------------------------------------------Content of load data tab
        # Number of epochs
        self.nepoch_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                         text="1. Number of epochs")
        self.nepoch_label.grid(row=0, column=0, sticky = "w")
        
        self.nepoch = ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="5")
               
        self.nepoch.grid(row=1, column=0, sticky = "w")
        self.nepoch.bind('<KeyRelease>', self.get_nepoch)
        CTkToolTip(self.nepoch, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message='Number of training epochs. Input should be numeric ') 
        
        # Learning rate
        self.learning_rate_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                                text="2. Learning rate")
        self.learning_rate_label.grid(row=2, column=0, sticky = "w")
        self.learning_rate= ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="0.01")
        self.learning_rate.grid(row=3, column=0, sticky = "w")
        self.learning_rate.bind('<KeyRelease>', self.get_learning_rate)
        
        # Warm up length
        self.warmup_length_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                         text="3. Warm-up length")
        self.warmup_length_label.grid(row=4, column=0, sticky = "w")
        self.warmup_length= ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="30") 
        self.warmup_length.grid(row=5, column=0, sticky = "w")
        self.warmup_length.bind('<KeyRelease>', self.get_warmup_length)
        CTkToolTip(self.warmup_length, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message='The number of timesteps used for warm-up. \n' +
                   'For example, the first n simulated outputs will be skipped \n'+
                   'when calculating the loss function. This value MUST \n'+
                   'be smaller than sequence length') 
        
        # Sequence length
        self.sequence_length_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                         text="4. Sequence length")
        self.sequence_length_label.grid(row=6, column=0, sticky = "w")
        self.sequence_length= ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="720") 
        self.sequence_length.grid(row=7, column=0, sticky = "w")
        self.sequence_length.bind('<KeyRelease>', self.get_sequence_length)
        CTkToolTip(self.sequence_length, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="The number of timesteps in each sample dataset. \n" + 
                   "One epoch loops over multiple baches. \n" +
                   "One batch consists of 'batch_size' sample datasets. \n" +
                   "One sample dataset consist of pairs of Input and Tartget \n" +
                   "    of 'sequence_length' timesteps (chronological order)")

        # Batch size
        self.batch_size_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                         text="5. Batch size")
        self.batch_size_label.grid(row=8, column=0, sticky = "w")
        self.batch_size= ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="3") 
        self.batch_size.grid(row=9, column=0, sticky = "w")
        self.batch_size.bind('<KeyRelease>', self.get_batch_size)
        CTkToolTip(self.batch_size, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="Please see sequence length for help") 
        
        # Patience
        self.patience_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                         text="6. Patience length")
        self.patience_label.grid(row=0, column=2, sticky = "w")
        self.patience= ctk.CTkEntry(master=self.tabview.tab("Trainer"),
                             placeholder_text="20") 
        self.patience.grid(row=1, column=2, sticky = "w")
        self.patience.bind('<KeyRelease>', self.get_patience_length)
        CTkToolTip(self.patience, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="The number of epochs to wait (before stopping) to \n" + 
                   "see if there is no improvement in ths validation loss, \n" + 
                   "which is used for early stopping. More information \n" + 
                   "please see https://github.com/Bjarten/early-stopping-pytorch. \n" +
                   "Patience length should be much smaller than the number of epochs")

        # Optimization method
        self.optim_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                        text="7. Optimization method")
        self.optim_label.grid(row=2, column=2, sticky = "w")    
        self.optim = ctk.CTkOptionMenu(self.tabview.tab("Trainer"),
                                                   values=['Adam'],
                                                   command=self.get_optim_method) 
        self.optim.grid(row=3, column=2, sticky = "w")
        
        # Loss function
        self.loss_function_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                               text="8. Loss function")
        self.loss_function_label.grid(row=4, column=2, sticky = "w")    
        self.loss = ctk.CTkOptionMenu(self.tabview.tab("Trainer"),
                                                   values=['Root Mean Square Error (RMSE)',
                                                           'Mean Absolute Error (MAE)',
                                                           'Mean Squared Error (MSE)'],
                                                   command=self.loss_function) 
        self.loss.grid(row=5, column=2, sticky = "w")
        
        # Save model
        self.out_dir_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                      text="9. Output directory")
        self.out_dir_label.grid(row=6, column=2, sticky = "w")     
        self.out_dir = ctk.CTkButton(self.tabview.tab("Trainer"), anchor='w', 
                                         command=self.out_dir_event,
                                         text="Select directory")
        self.out_dir.grid(row=7, column=2, sticky = "w")

        # Run model
        self.run_label = ctk.CTkLabel(self.tabview.tab("Trainer"), 
                                      text="10. Run model")
        self.run_label.grid(row=8, column=2, sticky = "w")     
        self.run = ctk.CTkButton(self.tabview.tab("Trainer"), anchor='w', 
                                         command=self.run_train_test,
                                         text="Run")
        self.run.grid(row=9, column=2, sticky = "w")
                
        # Progressbar
        self.progressbar = ctk.CTkProgressBar(master=self.tabview.tab("Trainer"))
        self.progressbar.grid(row=11, column=2,  sticky = "w", pady = (10,10))
        CTkToolTip(self.progressbar, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message="This is the training progress bar")
        self.progressbar.configure(mode="determinate", progress_color="orange")
        self.progressbar.set(0)
        
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
     
    # Get learning_rate
    def get_sequence_length(self, dummy):
        try:
            self.config["sequence_length"]  = int(self.sequence_length.get().strip())
            print(f"Sequence length = {self.config['sequence_length']}")
        except:
            tk.messagebox.showinfo(title="Error", message="Input should be integer")

    # Get learning_rate
    def get_batch_size(self, dummy):
        try:
            self.config["batch_size"]  = int(self.batch_size.get().strip())
            print(f"Batch size = {self.config['batch_size']}")
        except:
            tk.messagebox.showinfo(title="Error", message="Input should be integer")

    # Get learning_rate
    def get_patience_length(self, dummy):
        try:
            self.config["patience"]  = int(self.patience.get().strip())
            print(f"Patience = {self.config['patience']}")
        except:
            tk.messagebox.showinfo(title="Error", message="Input should be integer")
        
    # Get number of lstm layers
    def loss_function(self, method: str):
        loss_fn = {'Root Mean Square Error (RMSE)': "RMSE",
                    'Mean Absolute Error (MAE)': "MAE",
                    'Mean Squared Error (MSE)': "MSE"} 
                              
        self.config["loss_function"] = loss_fn[method]
        print(self.config["loss_function"])

    # Get number of lstm layers
    def get_optim_method(self, method: str):
        self.config["optim_method"] = method
        print(self.config["optim_method"])
        
    def out_dir_event(self):
        output_directory = tk.filedialog.askdirectory()
        self.config["output_directory"] = [output_directory]
        print("Output dir = ", self.config["output_directory"])

     
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
            print("done create model")
            print(self.config)
            # Train the model
            self.globalData["trainer"] = Trainer(config=self.config, 
                                               model=self.globalData["model"])
            print("done initial trainer")
            self.globalData["model"] = self.globalData["trainer"].train(
                self.globalData["x_train_scale"],
                self.globalData["y_train_scale"],
                self.globalData["x_valid_scale"],
                self.globalData["y_valid_scale"])
            print("done train model")
            # Run forward test the model
            y_train_simulated_scale = self.globalData["model"].evaluate(
                self.globalData["x_train_scale"])
            y_valid_simulated_scale = self.globalData["model"].evaluate(
                self.globalData["x_valid_scale"])
            y_test_simulated_scale = self.globalData["model"].evaluate(
                self.globalData["x_test_scale"])

            # Inverse scale/transform back simulated result to real scale
            self.globalData["y_train_simulated"] =\
                self.globalData["y_scaler"].inverse(y_train_simulated_scale)
            self.globalData["y_valid_simulated"] =\
                self.globalData["y_scaler"].inverse(y_valid_simulated_scale)
            self.globalData["y_test_simulated"] =\
                self.globalData["y_scaler"].inverse(y_test_simulated_scale)
                
            tk.messagebox.showinfo(title="Message box",
                                   message="Finished training/testing")
        except:
            None

        self.progressbar.set(1.0)        
        self.run.configure(state="normal")
        self.run.configure(fg_color=['#3a7ebf', '#1f538d'])

    
        
        
        