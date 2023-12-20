
import customtkinter as ctk

from .data_frame import DataFrame
from .network_design_frame import NetworkDesignFrame
from .project_summary_frame import ProjectSummaryFrame
from .sidebar_frame import SidebarFrame
from .train_test_frame import TrainTestFrame
from .visualize_frame import VisualizeFrame


class MainGUI(ctk.CTk):
    def __init__(self):

        # Initialize the interface apprearence
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("dark-blue") 

        # Initialize project setting
        global config
        global globalData
        
        config = {}
        config["scaler_input_dynamic_features"] = ["MinMaxScaler"]
        config["scaler_input_static_features"] = ["MinMaxScaler"]
        config["scaler_target_features"] = ["MinMaxScaler"]
        config["hidden_size"] = 30
        config["num_layers"] = 1
        config["activation_function_name"] = "Identity"
        config["n_epochs"] = 50
        config["learning_rate"] = 0.01
        config["dropout"] = 0.30
        config["warmup_length"] = 20

        # Initialize global data
        globalData = {}
        globalData['dynamic_data_header'] =[]
        globalData['static_data_header'] =[]
        

        
        super().__init__()
        
        #------------------------------------------------------configure window
        self.title("Hydro-ecological modelling with Long Short-Term Memory neural network")
        self.geometry(f"{1100}x{580}")
    
        #--------------------------------------------------------configure grid
        self.grid_columnconfigure(0, weight=0) 
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=0)
        self.grid_rowconfigure((0,1,2,3), weight=1)
    
        #----------------------------------------------------create main frames
        # data frame
        self.data_frame = DataFrame(container=self, config=config, 
                                    globalData=globalData)
        self.data_frame.grid(row=0, column=1, rowspan=4, columnspan=2,
                             padx = 10, pady=(20, 20), sticky="nsew")
    
        # network frame
        self.network_frame = NetworkDesignFrame(container=self, config=config,
                                                globalData=globalData)
            
        # train test frame
        self.train_test_frame = TrainTestFrame(container=self, config=config,
                                               globalData=globalData)
    
        # visualize frame
        self.visual_frame = VisualizeFrame(container=self, config=config,
                                           globalData=globalData)
            
        # summary frame
        self.summary_frame = ProjectSummaryFrame(container=self, config=config)
        self.summary_frame.grid(row=0, column=3, rowspan=4, padx = 10,
                                pady=(20, 20), sticky="nsew") 
            
        # sidebar frame - main frame controlling other frame
        self.sidebar_frame = SidebarFrame(container=self, 
                                          data_frame=self.data_frame,
                                          network_frame=self.network_frame,
                                          train_test_frame=self.train_test_frame,
                                          visual_frame=self.visual_frame)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, padx = 0,
                                pady=(20, 20), sticky="nsew") 

                 
# Display GUI 
def show_gui():
    app = MainGUI()
    app.mainloop()  