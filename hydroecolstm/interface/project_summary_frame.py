
import customtkinter as ctk
from CTkToolTip import CTkToolTip
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
                   message= 'Click here to update the project summary. \n' + 
                   'You could copy and save this to config.yml file for \n' +
                   'running hydroecolstm without the graphical user interface') 
        
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
                                        text = "Save as .yml", 
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
        try:
            file_name = ctk.filedialog.asksaveasfilename(
                title="Save project summary as .yml file",
                filetypes=(('yml files', '*.yml'),
                           ('All files', '*.*')))
            write_yml_file(config=self.config, out_file=file_name)
            print("Saved project summary as ", file_name)
        except:
            print("Error: Cannot save project summary")            
            
