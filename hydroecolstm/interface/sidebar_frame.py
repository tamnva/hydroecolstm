
import customtkinter as ctk
from CTkToolTip import CTkToolTip

class SidebarFrame(ctk.CTkFrame):
    def __init__(self, container=None, data_frame=None, network_frame=None,
                 train_test_frame=None, visual_frame=None, application_frame=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.data_frame = data_frame
        self.network_frame = network_frame
        self.train_test_frame = train_test_frame
        self.visual_frame = visual_frame
        self.application_frame = application_frame
        self.columnconfigure(0, weight=1)
        self.rowconfigure((0,1,2,3,4,5,6,7,8), weight=0)
        self.__create_widgets() 
    
    # create widgets for sidebar frame
    def __create_widgets(self):     
        self.label = ctk.CTkLabel(self, 
                                  text="HydroEcoLSTM", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        self.label.grid(row=0, column=0, padx=25, pady=20)
        CTkToolTip(self.label, delay=0.1, bg_color = 'orange', 
                   text_color = 'black', anchor = 'w', 
                   message= 'ALWAYS follow the numbering scheme (1 >> 2 >> 3' +
                   ' and so on) to create your project')

        # Data processing
        self.data_button = ctk.CTkButton(self, anchor='w', 
                                         command=self.data_button_event,
                                         text="1. Data Processing")
        self.data_button.grid(row=1, column=0, padx=0, pady=10)
        CTkToolTip(self.data_button, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click this button for preparing your ' +
                   'data input and target data\n')
        
        # Network definition
        self.net_button = ctk.CTkButton(self, anchor='w', 
                                        command=self.network_button_event,
                                        text="2. Network Design")
        self.net_button.grid(row=2, column=0, padx=0, pady=10)
        CTkToolTip(self.net_button, delay=0.1, bg_color = 'orange', 
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to design your neural network architect')
        
        # Training and testing the network
        self.train_button = ctk.CTkButton(self,
                                          command=self.traintest_button_event,
                                          anchor='w', 
                                          text="3. Training/Testing")
        self.train_button.grid(row=3, column=0, padx=0, pady=10)
        CTkToolTip(self.train_button, delay=0.1, bg_color = 'orange', 
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to set parameters for trainning and testing your network')

        self.visualize_button = ctk.CTkButton(self,
                                          command=self.visualize_button_event,
                                          anchor='w', 
                                          text="4. Visualize results")
        self.visualize_button.grid(row=4, column=0, padx=0, pady=10)
        CTkToolTip(self.visualize_button, delay=0.1, bg_color = 'orange', 
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to visualize the model outputs')

        self.application_button = ctk.CTkButton(self,
                                          command=self.application_button_event,
                                          anchor='w', 
                                          text="5. Application")
        self.application_button.grid(row=5, column=0, padx=0, pady=10)
        CTkToolTip(self.application_button, delay=0.1, bg_color = 'orange', 
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to make a forward run for forcast, or \n' +
                       'simulation in ungauged basins using the trained network')
        
        # Appearance
        self.appearance_option = ctk.CTkOptionMenu(self,
                                                   values=["Light", "Dark", "System"],
                                                   button_color='gray', fg_color = 'gray',
                                                   command=self.appearance_event) 
        self.appearance_option.place(relx=0.5, rely=0.90, anchor="n")
        CTkToolTip(self.appearance_option, delay=0.1, bg_color = 'orange', 
                   text_color = 'black', anchor='w', 
                   message='Please select your background themes here')
        
        self.scaling_optionemenu = ctk.CTkOptionMenu(self, 
                                                     values=["110%", "75%", "100%", "125%", "150%"],
                                                     button_color='gray', fg_color = 'gray',
                                                     command=self.change_scaling_event)
        self.scaling_optionemenu.place(relx=0.5, rely=0.82, anchor="n")
        CTkToolTip(self.scaling_optionemenu, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Please select the text size here \n')
        
        # Only activate the data_button when open GUI
        self.data_button.configure(fg_color=['#3a7ebf', '#1f538d'])
        self.net_button.configure(fg_color='gray')
        self.train_button.configure(fg_color='gray')
        self.visualize_button.configure(fg_color='gray')
        self.application_button.configure(fg_color='gray')
     
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)
        
    # button event in sidebar frame 
    def appearance_event(self, new_appearance: str):
        ctk.set_appearance_mode(new_appearance)
        
    def data_button_event(self):
        # turn of other frames
        self.network_frame.grid_forget()
        self.train_test_frame.grid_forget()
        self.visual_frame.grid_forget()
        self.application_frame.grid_forget()
        
        # turn on data frame
        self.data_frame.grid(row=0, column=1, rowspan=4, padx = 10, 
                               columnspan=2, pady=(20, 20), sticky="nsew")
        
        # change button color
        self.data_button.configure(fg_color=['#3a7ebf', '#1f538d'])
        self.net_button.configure(fg_color='gray')
        self.train_button.configure(fg_color='gray')
        self.visualize_button.configure(fg_color='gray')
        self.application_button.configure(fg_color='gray')
        
    def network_button_event(self):
        # turn of other frames
        self.data_frame.grid_forget()
        self.train_test_frame.grid_forget()
        self.visual_frame.grid_forget()
        self.application_frame.grid_forget()
        
        # turn on data frame
        self.network_frame.grid(row=0, column=1, rowspan=4, padx = 10, 
                               columnspan=2, pady=(20, 20), sticky="nsew")
        
        # change button color
        self.data_button.configure(fg_color='gray')
        self.net_button.configure(fg_color=['#3a7ebf', '#1f538d'])
        self.train_button.configure(fg_color='gray')
        self.visualize_button.configure(fg_color='gray')
        self.application_button.configure(fg_color='gray')
        
    def traintest_button_event(self):
        # turn of other frames
        self.data_frame.grid_forget()
        self.network_frame.grid_forget()
        self.visual_frame.grid_forget()
        self.application_frame.grid_forget()
        
        # turn on data frame
        self.train_test_frame.grid(row=0, column=1, rowspan=4, padx = 10, 
                               columnspan=2, pady=(20, 20), sticky="nsew")
        
        # change button color
        self.data_button.configure(fg_color='gray')
        self.train_button.configure(fg_color=['#3a7ebf', '#1f538d'])
        self.net_button.configure(fg_color='gray')
        self.visualize_button.configure(fg_color='gray')
        self.application_button.configure(fg_color='gray')
        
    def visualize_button_event(self):
        # turn of other frames
        self.data_frame.grid_forget()
        self.network_frame.grid_forget()
        self.train_test_frame.grid_forget()
        self.application_frame.grid_forget()
        
        # turn on data frame
        self.visual_frame.grid(row=0, column=1, rowspan=4, padx = 10, 
                               columnspan=2, pady=(20, 20), sticky="nsew")
        
        # change button color
        self.data_button.configure(fg_color='gray')
        self.train_button.configure(fg_color='gray')
        self.net_button.configure(fg_color='gray')
        self.visualize_button.configure(fg_color=['#3a7ebf', '#1f538d'])
        self.application_button.configure(fg_color='gray')
        
    def application_button_event(self):
        self.data_frame.grid_forget()
        self.network_frame.grid_forget()
        self.train_test_frame.grid_forget()
        self.visual_frame.grid_forget()
   
        # turn on data frame
        self.application_frame.grid(row=0, column=1, rowspan=4, padx = 10, 
                               columnspan=2, pady=(20, 20), sticky="nsew")
        
        # change button color
        self.data_button.configure(fg_color='gray')
        self.train_button.configure(fg_color='gray')
        self.net_button.configure(fg_color='gray')
        self.visualize_button.configure(fg_color='gray')
        self.application_button.configure(fg_color=['#3a7ebf', '#1f538d'])
                
        
        
        