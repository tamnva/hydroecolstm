import tkinter as tk
import customtkinter as ctk
from hydroecolstm.model_run import run_config
from hydroecolstm.data.read_data import read_scale_data
from hydroecolstm.model.create_model import create_model
from hydroecolstm.train.trainer import Trainer
from hydroecolstm.model_run import config_to_search_space
from CTkToolTip import CTkToolTip
import torch
from ray import tune    # is used within the eval function

class TrainTestFrame(ctk.CTkScrollableFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config = config
        self.globalData = globalData
        self.__create_widgets() 
        
    # create widgets for sidebar frame 
    def __create_widgets(self): 
        
        # create tabs #
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5,
                                      fg_color = "transparent")
        self.tabview.pack(fill='both',expand=1)
        self.tabview.add("1. Initial state dicts")
        self.tabview.tab("1. Initial state dicts").grid_columnconfigure((0), weight=1)
        self.tabview.add("2. Trainer")
        self.tabview.tab("2. Trainer").grid_columnconfigure((0,1), weight=1)
        
        # Tooltip for tune seach space api
        self.ray_tooltip = ('Or input the tune Search Space API command, ' +
                            'e.g., tune.grid_search([20, 30]). Please see:' + 
                            'https://docs.ray.io/en/latest/tune/api/search_space.html')
  
        # ---------------------------------------------Initial state dicts      
        # Load model state dict
        self.load_model_label = ctk.CTkLabel(self.tabview.tab("1. Initial state dicts"), 
                                      text="1. Load model state dicts")
        self.load_model_label.grid(row=0, column=0, sticky = "w")  
        self.load_model = ctk.CTkButton(self.tabview.tab("1. Initial state dicts"), anchor='w', 
                                         command=self.load_state_dict,
                                         text="Load model")
        self.load_model.grid(row=1, column=0, sticky = "w")
        CTkToolTip(self.load_model, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="Optional input - load initial model state dicts. " +
                   "This could be, e.g., the calibrated model at a regional scale " +
                   "and used in here for parameter fine tuning. The model " +
                   "state dict is ONLY USED with the MANUAL hyperparamter " +
                   "optimization, incase of AUTOMATIC hyperparameter tuning "+
                   "with ray tune, this option is discarded")
        
        # ---------------------------------------------Content of load data tab
        # Number of epochs
        self.nepoch_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                         text="1. Number of epochs")
        self.nepoch_label.grid(row=0, column=0, sticky = "w")
        
        self.nepoch = ctk.CTkEntry(master=self.tabview.tab("2. Trainer"),
                             placeholder_text="5")
               
        self.nepoch.grid(row=1, column=0, sticky = "w")
        self.nepoch.bind('<KeyRelease>', self.get_nepoch)
        CTkToolTip(self.nepoch, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message='Number of training epochs (positive integer number)') 
        
        # Learning rate
        self.learning_rate_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                                text="2. Learning rate")
        self.learning_rate_label.grid(row=2, column=0, sticky = "w")
        self.learning_rate= ctk.CTkEntry(master=self.tabview.tab("2. Trainer"),
                             placeholder_text="0.01")
        self.learning_rate.grid(row=3, column=0, sticky = "w")
        self.learning_rate.bind('<KeyRelease>', self.get_learning_rate)
        CTkToolTip(self.learning_rate, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message='Learning rate for gradient descent (positive real number)') 
        
        # Warm up length
        self.warmup_length_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                         text="3. Warm-up length")
        self.warmup_length_label.grid(row=4, column=0, sticky = "w")
        self.warmup_length= ctk.CTkEntry(master=self.tabview.tab("2. Trainer"),
                             placeholder_text="30") 
        self.warmup_length.grid(row=5, column=0, sticky = "w")
        self.warmup_length.bind('<KeyRelease>', self.get_warmup_length)
        CTkToolTip(self.warmup_length, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message='The number of timesteps used for warm-up. \n' +
                   'For example, the first warmup_length outputs will be skipped \n'+
                   'when calculating the loss function. This value MUST \n'+
                   'be smaller than sequence length') 
        
        # Sequence length
        self.sequence_length_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                         text="4. Sequence length")
        self.sequence_length_label.grid(row=6, column=0, sticky = "w")
        self.sequence_length= ctk.CTkEntry(master=self.tabview.tab("2. Trainer"),
                             placeholder_text="720") 
        self.sequence_length.grid(row=7, column=0, sticky = "w")
        self.sequence_length.bind('<KeyRelease>', self.get_sequence_length)
        CTkToolTip(self.sequence_length, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="The number of timesteps in each sample dataset. \n" + 
                   "One epoch loops over multiple baches. \n" +
                   "One batch consists of 'batch_size' sample datasets. \n" +
                   "One sample dataset consist of pairs of Input and Tartget \n" +
                   "    of 'sequence_length' timesteps (chronological order)")

        # Batch size
        self.batch_size_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                         text="5. Batch size")
        self.batch_size_label.grid(row=8, column=0, sticky = "w")
        self.batch_size= ctk.CTkEntry(master=self.tabview.tab("2. Trainer"),
                             placeholder_text="3") 
        self.batch_size.grid(row=9, column=0, sticky = "w")
        self.batch_size.bind('<KeyRelease>', self.get_batch_size)
        CTkToolTip(self.batch_size, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="Please see sequence length for help") 
        
        # Patience
        self.patience_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                         text="6. Patience length")
        self.patience_label.grid(row=0, column=2, sticky = "w")
        self.patience= ctk.CTkEntry(master=self.tabview.tab("2. Trainer"),
                             placeholder_text="20") 
        self.patience.grid(row=1, column=2, sticky = "w")
        self.patience.bind('<KeyRelease>', self.get_patience_length)
        CTkToolTip(self.patience, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="The number of epochs to wait (before stopping) to \n" + 
                   "see if there is no improvement in ths validation loss, \n" + 
                   "which is used for early stopping. More information \n" + 
                   "please see https://github.com/Bjarten/early-stopping-pytorch. \n" +
                   "Patience length should be much smaller than the number of epochs")

        # Optimization method
        self.optim_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                        text="7. Optimization method")
        self.optim_label.grid(row=2, column=2, sticky = "w")    
        self.optim = ctk.CTkOptionMenu(self.tabview.tab("2. Trainer"),
                                                   values=['Adam'],
                                                   command=self.get_optim_method) 
        self.optim.grid(row=3, column=2, sticky = "w")
        CTkToolTip(self.optim, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="In this version only Adam method is available")
        
        # Loss function
        self.loss_function_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                               text="8. Loss function")
        self.loss_function_label.grid(row=4, column=2, sticky = "w")    
        self.loss = ctk.CTkOptionMenu(self.tabview.tab("2. Trainer"),
                                                   values=['Root Mean Square Error',
                                                           'Mean Absolute Error',
                                                           'Mean Squared Error',
                                                           '1 - Nash Sutcliffe'],
                                                   command=self.loss_function) 
        self.loss.grid(row=5, column=2, sticky = "w")
        CTkToolTip(self.loss, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="Please select the loss function for calculating"+
                   " traning and validatation losses, which are used for" +
                   " updating model paramters and early stopping")
        
        # Save model
        self.out_dir_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                      text="9. Output directory")
        self.out_dir_label.grid(row=6, column=2, sticky = "w")     
        self.out_dir = ctk.CTkButton(self.tabview.tab("2. Trainer"), anchor='w', 
                                         command=self.out_dir_event,
                                         text="Select directory")
        self.out_dir.grid(row=7, column=2, sticky = "w")
        CTkToolTip(self.out_dir, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="Please select the directory for saving model outputs"+
                   " If not selected, the default directory (pathlib.Path.cwd()) is used")
        
        
        # Run model
        self.run_label = ctk.CTkLabel(self.tabview.tab("2. Trainer"), 
                                      text="10. Run model")
        self.run_label.grid(row=8, column=2, sticky = "w")     
        self.run = ctk.CTkButton(self.tabview.tab("2. Trainer"), anchor='w', 
                                         command=self.run_train_test,
                                         text="Run")
        self.run.grid(row=9, column=2, sticky = "w")
        CTkToolTip(self.run, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="Click here to run the model. While model is running"+
                   " in the background, the GUI will be frozen.")
                
        # Progressbar
        self.progressbar = ctk.CTkProgressBar(master=self.tabview.tab("2. Trainer"))
        self.progressbar.grid(row=10, column=2,  sticky = "w", pady = (10,10))
        CTkToolTip(self.progressbar, delay=0.1, bg_color = 'orange', justify = "left",
                   text_color = 'black', anchor='w',  wraplength=250, 
                   message="This is the training progress bar." +
                   " It will turn to full orange when the training was completed")
        
        self.progressbar.configure(mode="determinate", progress_color="orange")
        self.progressbar.set(0)
        
    # Get number of epochs
    def get_nepoch(self, dummy):
        # Get number of hidden layers
        get_input_text = self.nepoch.get().strip()
        
        try:
            self.config["n_epochs"] = int(float(get_input_text))
            print("Number of epochs = ", self.config["n_epochs"])
            self.nepoch.configure(fg_color = '#d3ffc7')
        except:
            try:
                eval(get_input_text)
                self.config["n_epochs"] = get_input_text
                print("Number of epochs = ", self.config["n_epochs"])
                self.nepoch.configure(fg_color = '#d3ffc7')
            except:
                self.nepoch.configure(fg_color = '#ffc7c7')
                
    # Get learning_rate
    def get_learning_rate(self, dummy):
        
        get_input_text = self.learning_rate.get().strip()
        
        try:
            self.config["learning_rate"] = float(get_input_text)
            print("learning rate = ", self.config["learning_rate"])
            self.learning_rate.configure(fg_color = '#d3ffc7')
        except:
            try:
                eval(get_input_text)
                self.config["learning_rate"] = get_input_text
                print("learning rate = ", self.config["learning_rate"])
                self.learning_rate.configure(fg_color = '#d3ffc7')
            except:
                self.learning_rate.configure(fg_color = '#ffc7c7')
                

    # Get warm up length
    def get_warmup_length(self, dummy):

        get_input_text = self.warmup_length.get().strip()
        
        try:
            self.config["warmup_length"] = int(float(get_input_text))
            print("Warm up length = ", self.config["warmup_length"])
            self.warmup_length.configure(fg_color = '#d3ffc7')
        except:
            try:
                eval(get_input_text)
                self.config["warmup_length"] = get_input_text
                print("learning rate = ", self.config["warmup_length"])
                self.warmup_length.configure(fg_color = '#d3ffc7')
            except:
                self.warmup_length.configure(fg_color = '#ffc7c7')
                
                
    # Get learning_rate
    def get_sequence_length(self, dummy):
        
        get_input_text = self.sequence_length.get().strip()
        
        try:
            self.config["sequence_length"] = int(float(get_input_text))
            print("sequence length = ", self.config["sequence_length"])
            self.sequence_length.configure(fg_color = '#d3ffc7')
        except:
            try:
                eval(get_input_text)
                self.config["sequence_length"] = get_input_text
                print("learning rate = ", self.config["sequence_length"])
                self.sequence_length.configure(fg_color = '#d3ffc7')
            except:
                self.sequence_length.configure(fg_color = '#ffc7c7')
                

    # Get learning_rate
    def get_batch_size(self, dummy):
        
        get_input_text = self.batch_size.get().strip()
        
        try:
            self.config["batch_size"] = int(float(get_input_text))
            print("sequence length = ", self.config["batch_size"])
            self.batch_size.configure(fg_color = '#d3ffc7')
        except:
            try:
                eval(get_input_text)
                print("batch size = ", self.config["batch_size"])
                self.batch_size.configure(fg_color = '#d3ffc7')
            except:
                self.batch_size.configure(fg_color = '#ffc7c7')
                

    # Get learning_rate
    def get_patience_length(self, dummy):
        
        get_input_text = self.patience.get().strip()
        
        try:
            self.config["patience"] = int(float(get_input_text))
            print("sequence length = ", self.config["patience"])
            self.patience.configure(fg_color = '#d3ffc7')
        except:
            try:
                eval(get_input_text)
                self.config["patience"] = get_input_text
                print("patience length = ", self.config["patience"])
                self.patience.configure(fg_color = '#d3ffc7')
            except:
                self.patience.configure(fg_color = '#ffc7c7')
                
    # Get number of lstm layers
    def loss_function(self, method: str):
        loss_fn = {'Root Mean Square Error': "RMSE",
                   'Mean Absolute Error': "MAE",
                   'Mean Squared Error': "MSE",
                   '1 - Nash Sutcliffe': "NSE_complement"} 
                              
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
        
        # update text of the selected directory
        if len(output_directory) > 0:
            self.out_dir.configure(text = "..." + output_directory[-30:])
        else:
            self.out_dir.configure(text = "Select directory")
    
    def load_state_dict(self):
        # Select model state dict file
        state_dict_file = ctk.filedialog.askopenfilename(title="Select model state dict file", 
                                              filetypes=(('pt files', '*.pt'),
                                                         ('All files', '*.*')))
        # update text load model
        if state_dict_file[-3:] == ".pt":
            self.load_model.configure(text = "..." + state_dict_file[-30:])
            self.globalData['init_state_dicts'] = True
            self.globalData['init_state_dicts_file'] = state_dict_file
        else:
            self.load_model.configure(text = "Load model")
            self.globalData['init_state_dicts'] = False

    # Message box for error in input with tune search space 
    def message_box_tune(self, data_type):
        tk.messagebox.showinfo(
            title="Error",
            message="Input should be " + data_type + " or Tune Search Space API command: " +
            "https://docs.ray.io/en/latest/tune/api/search_space.html")

     
    def run_train_test(self):
        # Set progress to zero
        self.progressbar.set(0)

        self.run.configure(fg_color='gray')
        self.run.configure(state="disabled")
        
        tk.messagebox.showinfo(title="Message box", 
                               message="Trainning will start after closing this box")
        
        
        try:
            
            # Read data
            data = read_scale_data(self.config)
            
            # Convert config to search space
            search_space = config_to_search_space(
                self.config,
                data['x_train_scale'],
                data['y_train_scale'],
                data['x_valid_scale'],
                data['y_valid_scale']
                )
            
            if self.globalData['init_state_dicts'] and search_space["is_manual_optim"]:
                # Create the model
                model = create_model(self.config)
                
                model.load_state_dict(
                    torch.load(
                        self.globalData['init_state_dicts_file']
                        )
                    )
                
                # Train with train dataset
                trainer = Trainer(self.config, model)
                model = trainer.train(
                    data['x_train_scale'], 
                    data['y_train_scale'],
                    data['x_valid_scale'], 
                    data['y_valid_scale'])
        
                # Save train loss per epoch and best train loss
                data["loss_epoch"] = trainer.loss_epoch
                data["best_train_loss"] = trainer.best_train_loss
                best_config = self.config

            else:
                model, data, best_config = run_config(self.config)

            # Evaluate the model and transform back to normal scale
            data['y_train_simulated'] = data["y_scaler"].inverse(
                model.evaluate(data["x_train_scale"])
                )
            
            data['y_valid_simulated'] = data["y_scaler"].inverse(
                model.evaluate(data["x_valid_scale"])
                )
            
            data['y_test_simulated'] = data["y_scaler"].inverse(
                model.evaluate(data["x_test_scale"])
                )

            # Update global data
            self.globalData.update(data)
            self.globalData["model"] = model
            self.globalData["best_config"] = best_config
            self.globalData["config"] = self.config
            
 
            tk.messagebox.showinfo(title="Message box",
                                   message="Finished training/testing")
        except:
            tk.messagebox.showinfo(title="Error",
                                   message="Cannot train the model")

        self.progressbar.set(1.0)        
        self.run.configure(state="normal")
        self.run.configure(fg_color=['#3a7ebf', '#1f538d'])

    
        
        
        