
import customtkinter as ctk
import tkinter as tk
from CTkToolTip import CTkToolTip
from CTkMessagebox import CTkMessagebox
import torch
from pathlib import Path

from hydroecolstm.interface.utility import config_to_text, sort_key
from hydroecolstm.interface.utility import write_yml_file

class ProjectSummaryFrame(ctk.CTkFrame):
    def __init__(self, container=None, config=None):
        super().__init__(container)
        self.config = config
        # setup the grid layout manager
        self.columnconfigure(0, weight=1)
        self.rowconfigure((0), weight=0)
        self.rowconfigure((1), weight=1)
        self.rowconfigure((2), weight=0)
        self.__create_widgets() 
        
    # create widgets for sidebar frame
    def __create_widgets(self):     
        self.update_summary = ctk.CTkButton(self, text="Project Summary", 
                                            font=ctk.CTkFont(size=20, weight="bold"),
                                            command=self.update_project_summary, 
                                            fg_color = "transparent",
                                            text_color="black")
        self.update_summary.grid(row=0, column=0, pady=0, padx=0)
        CTkToolTip(self.update_summary, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to update the project summary') 
        
        self.summary_textbox = ctk.CTkTextbox(master=self,corner_radius=0, 
                                              height=2000,
                                              bg_color='transparent', 
                                              fg_color='transparent',
                                              activate_scrollbars=True,
                                              wrap='none')

        self.summary_textbox.grid(row=1, column=0,pady=(10,7), padx=0)
        self.summary_textbox.insert("end", "Click 'Project Summary'\n" )
        self.summary_textbox.insert("end", "to see project info: " )
        self.summary_textbox.configure(spacing3=10) 

        self.save_buton = ctk.CTkButton(self, border_color="grey",
                                        border_width=1.5,
                                        command=self.save_yml,
                                        text = "Save", 
                                        fg_color = "transparent",
                                        text_color="black")
        CTkToolTip(self.save_buton, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to save project summary as' +
                   ' configuration file to run later without graphical interface')

        self.save_buton.grid(row=2, column=0,pady=(10,7), padx=0)
        
    def update_project_summary(self):
        # Delete text
        self.summary_textbox.delete("0.0", "end")
        output_text = config_to_text(config=sort_key(self.config))
    
        for text in output_text:
            self.summary_textbox.insert("end", text)
    

    def save_yml(self):
        
        # Ask which one user want to save
        msg = CTkMessagebox(title="Save", message="Please select save option",
                    option_1="Cancel", 
                    option_3="Save all",
                    option_2="Save config")
        response = msg.get()

        try:
            if response == "Save config":
                
                # Save config as .yml
                file_name = ctk.filedialog.asksaveasfilename(
                    title="Save project summary as .yml file",
                    filetypes=(('yml files', '*.yml'),
                               ('All files', '*.*')))
                write_yml_file(config=self.config, out_file=file_name)
                print("Saved project summary as ", file_name)
                
            elif response == "Save all":
                # Save config as .yml             
                output_dir = tk.filedialog.askdirectory()
                self.config["output_dir"] = [output_dir]
                print("Saved project summary as config.yml file")
                
                write_yml_file(config=self.config, out_file=
                               Path(self.config["output_dir"][0], "config.yml"))
        
                # Save model_state_dicts to model_state_dict.pt file
                torch.save(self.globalData["model"].state_dict(), 
                           Path(self.config["output_dir"][0], "model_state_dict.pt"))
                print("Model state_dict was saved as model_state_dict.pt")
                
                # Save global data
                torch.save(self.globalData, 
                           Path(self.config["output_dir"][0], "globalData.pt"))
                print("globalData was saved as globalData.pt")
            else:
                pass
                
        except:
            print("Error: Cannot save model or data")            
            
