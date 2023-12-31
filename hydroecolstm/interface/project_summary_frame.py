
import customtkinter as ctk
from CTkToolTip import CTkToolTip
from hydroecolstm.interface.utility import config_to_text, sort_key

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
        CTkToolTip(self.update_summary, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to update the project summary. \n' + 
                   'You could copy and save this to config.yml file for \n' +
                   'running hydroecolstm without the graphical user interface') 
        

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
        #print(self.config)
        output_text = config_to_text(config=sort_key(self.config))
    
        for text in output_text:
            self.summary_textbox.insert("end", text)

        