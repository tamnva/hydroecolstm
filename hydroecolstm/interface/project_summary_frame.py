
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
        output_text = self.config_to_list_text(config=self.config)
    
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
                    if (len(config[key]) == 2):
                        out_text.append(key +": \n")
                        out_text.append("  - " + str(config["train_period"][0])[:16] + "\n")
                        out_text.append("  - " + str(config["train_period"][1])[:16] + "\n")
                except:
                    out_text.append(key +": " + str(config[key]) + "\n")
        return out_text
  

        