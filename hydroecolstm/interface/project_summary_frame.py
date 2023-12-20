
import customtkinter as ctk
import yaml

class ProjectSummaryFrame(ctk.CTkScrollableFrame):
    def __init__(self, container=None, config=None):
        super().__init__(container)
        self.config = config
        # setup the grid layout manager
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        #self.rowconfigure((0,1,2,3,4,5,6,7), weight=0)
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self):     
        self.update_summary = ctk.CTkButton(self, text="Project Summary", 
                                  font=ctk.CTkFont(size=20, weight="bold"),
                                  command=self.update_project_summary, 
                                  fg_color = "transparent",
                                  text_color="black")
        self.update_summary.pack(pady=0, padx=0)

        self.summary_textbox = ctk.CTkTextbox(master=self,corner_radius=0, 
                                              height=2500, 
                                              bg_color='transparent', 
                                              fg_color='transparent',
                                              activate_scrollbars=True,
                                              wrap='none')

        self.summary_textbox.pack(pady=10, padx=0)
        
        self.summary_textbox.insert("end", "Click 'Project Summary'\n" )
        self.summary_textbox.insert("end", "to see project info: " )
        self.summary_textbox.configure(spacing3=10) 

    def update_project_summary(self):
        # Delete text
        self.summary_textbox.delete("0.0", "end")
        
        # update selected dynamic data file name
        try:
            if "dynamic_data_file" in self.config.keys():
                self.summary_textbox.insert("end", "dynamic_data_file:\n" )
                self.summary_textbox.insert("end", "  - " + 
                                            self.config["dynamic_data_file"][0] + "\n")
        except:
            None

        # update selected static data file name
        try:        
            if "static_data_file" in self.config.keys():
                self.summary_textbox.insert("end", "static_data_file:\n" )
                self.summary_textbox.insert("end", "  - " + 
                                            self.config["static_data_file"][0] + "\n")
        except:
            None
               
        if "object_id" in self.config.keys():
            self.summary_textbox.insert("end", "object_id:\n")
            for key in self.config["object_id"]:
                self.summary_textbox.insert("end", "  - " + str(key) + "\n")
            
        if "input_dynamic_features" in self.config.keys():
            self.summary_textbox.insert("end", "input_dynamic_features:\n")
            for key in self.config["input_dynamic_features"]:
                self.summary_textbox.insert("end", "  - " + str(key) + "\n")

        if "input_static_features" in self.config.keys():
            self.summary_textbox.insert("end", "input_static_features:\n")
            for key in self.config["input_static_features"]:
                self.summary_textbox.insert("end", "  - " + str(key) + "\n")
                
        if "target_features" in self.config.keys():
            self.summary_textbox.insert("end", "target_features:\n")
            for key in self.config["target_features"]:
                self.summary_textbox.insert("end", "  - " + str(key) + "\n")

        if "target_features" in self.config.keys():
            self.summary_textbox.insert("end", "target_features:\n")
            for key in self.config["target_features"]:
                self.summary_textbox.insert("end", "  - " + str(key) + "\n")

        if "scaler_input_dynamic_features" in self.config.keys():
            self.summary_textbox.insert("end", "scaler_input_dynamic_features:\n")
            self.summary_textbox.insert("end", "  - " + 
                                        self.config["scaler_input_dynamic_features"][0] + "\n")

        if "scaler_input_static_features" in self.config.keys():
            self.summary_textbox.insert("end", "scaler_input_static_features:\n")
            self.summary_textbox.insert("end", "  - " + 
                                        self.config["scaler_input_static_features"][0] + "\n")

        if "train_period" in self.config.keys():
            self.summary_textbox.insert("end", "train_period:\n")
            for key in self.config["train_period"]:
                self.summary_textbox.insert("end", "  - " + str(key) + "\n")
                
        if "test_period" in self.config.keys():
            self.summary_textbox.insert("end", "test_period:\n")
            for key in self.config["test_period"]:
                self.summary_textbox.insert("end", "  - " + str(key) + "\n")
 
        if "hidden_size" in self.config.keys():
            self.summary_textbox.insert("end", "hidden_size:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["hidden_size"]) + "\n")               

        if "num_layers" in self.config.keys():
            self.summary_textbox.insert("end", "num_layers:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["num_layers"]) + "\n")               

        if "dropout" in self.config.keys():
            self.summary_textbox.insert("end", "dropout:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["dropout"]) + "\n")      
                
        if "activation_function_name" in self.config.keys():
            self.summary_textbox.insert("end", "activation_function_name:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["activation_function_name"]) + "\n")                     

        if "n_epochs" in self.config.keys():
            self.summary_textbox.insert("end", "n_epochs:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["n_epochs"]) + "\n")    

        if "learning_rate" in self.config.keys():
            self.summary_textbox.insert("end", "learning_rate:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["learning_rate"]) + "\n")
            
        if "warmup_length" in self.config.keys():
            self.summary_textbox.insert("end", "warmup_length:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["warmup_length"]) + "\n")               

        if "objective_function_name" in self.config.keys():
            self.summary_textbox.insert("end", "objective_function_name:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["objective_function_name"]) + "\n")     

        if "optim_method" in self.config.keys():
            self.summary_textbox.insert("end", "optim_method:\n")
            self.summary_textbox.insert("end", "  - " + str(self.config["optim_method"]) + "\n")    
        
        
        