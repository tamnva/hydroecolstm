
import customtkinter as ctk

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
        output_text = self.config_to_list_text(config=self.sort_key(self.config))
    
        for text in output_text:
            self.summary_textbox.insert("end", text)
 
    def config_to_list_text(self, config):
        
        out_text = []
        for key in config.keys():                
            if type(config[key]) is list:
                
                out_text.append(key + ":\n")
                for element in config[key]:
                    out_text.append("  - " + str(element) + "\n")
            else:
                try:
                    if (len(config[key]) > 0):
                        out_text.append(key +": \n")
                        out_text.append("  - " + str(config["train_period"][0])[:16] + "\n")
                        out_text.append("  - " + str(config["train_period"][1])[:16] + "\n")
                except:
                    out_text.append(key +": " + str(config[key]) + "\n")
        return out_text
    
    def sort_key(self, config):
        
        config_sort = {}
        
        if "dynamic_data_file" in config.keys():
            config_sort["dynamic_data_file"] = config["dynamic_data_file"]

        if "static_data_file" in config.keys():
            config_sort["static_data_file"] = config["static_data_file"]

        if "input_static_features" in config.keys():
            config_sort["input_static_features"] = config["input_static_features"] 

        if "input_dynamic_features" in config.keys():
            config_sort["input_dynamic_features"] = config["input_dynamic_features"]        

        if "target_features" in config.keys():
            config_sort["target_features"] = config["target_features"] 

        if "object_id" in config.keys():
            config_sort["object_id"] = config["object_id"]        

        if "train_period" in config.keys():
            config_sort["train_period"] = config["train_period"]

        if "test_period" in config.keys():
            config_sort["test_period"] = config["test_period"]

        if "scaler_input_dynamic_features" in config.keys():
            config_sort["scaler_input_dynamic_features"] = config["scaler_input_dynamic_features"]        

        if "scaler_input_static_features" in config.keys():
            config_sort["scaler_input_static_features"] = config["scaler_input_static_features"] 

        if "scaler_target_features" in config.keys():
            config_sort["scaler_target_features"] = config["scaler_target_features"]        

        if "hidden_size" in config.keys():
            config_sort["hidden_size"] = config["hidden_size"] 

        if "num_layers" in config.keys():
            config_sort["num_layers"] = config["num_layers"]        

        if "activation_function_name" in config.keys():
            config_sort["activation_function_name"] = config["activation_function_name"]

        if "n_epochs" in config.keys():
            config_sort["n_epochs"] = config["n_epochs"] 

        if "learning_rate" in config.keys():
            config_sort["learning_rate"] = config["learning_rate"]        

        if "dropout" in config.keys():
            config_sort["dropout"] = config["dropout"] 

        if "warmup_length" in config.keys():
            config_sort["warmup_length"] = config["warmup_length"]        

        if "optim_method" in config.keys():
            config_sort["optim_method"] = config["optim_method"] 

        if "objective_function_name" in config.keys():
            config_sort["objective_function_name"] = config["objective_function_name"]        

        if "output_dir" in config.keys():
            config_sort["output_dir"] = config["output_dir"]

        if "output_dir" in config.keys():
            config_sort["output_dir"] = config["output_dir"] 

        if "static_data_file_forecast" in config.keys():
            config_sort["static_data_file_forecast"] = config["static_data_file_forecast"]        

        if "forecast_period" in config.keys():
            config_sort["forecast_period"] = config["forecast_period"] 

        if "object_id_forecast" in config.keys():
            config_sort["object_id_forecast"] = config["object_id_forecast"]  
            
        return config_sort

        