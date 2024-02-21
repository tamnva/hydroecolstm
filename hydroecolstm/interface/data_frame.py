
import customtkinter as ctk
import tkinter as tk
import pandas as pd
from pandastable import Table
import tkcalendar as tkc
from CTkListbox import CTkListbox
from CTkToolTip import CTkToolTip
from hydroecolstm.data.scaler import Scaler, get_scaler_name
from hydroecolstm.data.read_data import read_train_valid_test_data
   
class DataFrame(ctk.CTkScrollableFrame):
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
        self.tabview.add("1. Load data")
        self.tabview.tab("1. Load data").grid_columnconfigure((1,1), weight=1)
        self.tabview.add("2. Filter data")
        self.tabview.tab("2. Filter data").grid_columnconfigure((1,1), weight=1)
        self.tabview.add("3. Transform data")
        self.tabview.tab("3. Transform data").grid_columnconfigure((1,1), weight=1)
        
        # ---------------------------------------------content of load data tab
        # ---------------------------------------------------------Dynamic data
        self.dynamic_label = ctk.CTkLabel(self.tabview.tab("1. Load data"), 
                                         text="1. Dynamic/time series data")
        self.dynamic_label.grid(row=0, column=0, padx = 10, pady=(5,5), sticky="w")
        
        # Select dynamic data label
        self.select_dynamic_file = ctk.CTkButton(self.tabview.tab("1. Load data"), 
                                               anchor='w', 
                                               text="Select dynamic data file", 
                                               command=self.get_dynamic_file)
        self.select_dynamic_file.grid(row=1, column=0, padx = 10, pady=(5,5), sticky="w")        
        CTkToolTip(self.select_dynamic_file, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='e',  wraplength=500, 
                   message='Click here to select dynamic (timeseries) data file. ' + 
                   'The data file must be in .csv format with header. ' +
                   'This data file must contains at least two colums, ' +
                   'one with the name <object_id> and the other column with the ' +
                   'name <time> in format yyyy-mm-dd hh:mm.The object_id column will '+ 
                   'link object in dynamic/timeseries data file with static data file. ' +
                   'Timeseries of BOTH input and target features must be in this file. ' +
                   'Template of this file can be seen in the package documentation')
        
        # display selected data file
        self.selected_dynamic_filename = ctk.CTkLabel(self.tabview.tab("1. Load data"),
                                                      text="No file was selected")
        self.selected_dynamic_filename.grid(row=2, column=0, padx = 10, pady=(5,5), sticky="w")
        
        #  display dynamic data label
        self.show_data_statistics = ctk.CTkButton(self.tabview.tab("1. Load data"), 
                                              anchor='e', 
                                              text="Display/visualize dynamic data", 
                                              command=self.display_dynamic_data)
        self.show_data_statistics.grid(row=3, column=0, padx = 10, 
                                   pady=(5,5), sticky="w")        
        
        
        # check box dynamic data
        self.check_dynamic_data = ctk.IntVar(value=0)         
        self.checkbox_dynamic_data = ctk.CTkCheckBox(self.tabview.tab("1. Load data"), 
                                               text="Check/uncheck to use/discard data",
                                               command=self.load_dynamic_data, 
                                               variable=self.check_dynamic_data)
        self.checkbox_dynamic_data.grid(row=4, column=0, padx = 10, pady=(10,10), sticky="w")
        CTkToolTip(self.checkbox_dynamic_data, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'If you check this box, data from the selected \n' +
                       'dynamic data file will be used, uncheck means data will \n' +
                       'not be used')

        # ---------------------------------------------------------Static data       
        self.static_label = ctk.CTkLabel(self.tabview.tab("1. Load data"), 
                                        text="2. Static attribute data")
        self.static_label.grid(row=0, column=2, padx = 10, pady=(5,5), sticky="w")
        
        # select static data file
        self.select_static_file = ctk.CTkButton(self.tabview.tab("1. Load data"), 
                                              anchor='e', 
                                              text="Select static data file", 
                                              command=self.get_static_file)

        self.select_static_file.grid(row=1, column=2, padx = 10, 
                                   pady=(5,5), sticky="w")
        CTkToolTip(self.select_static_file, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message='Click here to select static (catchment attributes) data file. ' + 
                   'the data file must be in .csv format with header. ' + 
                   'This data file must contains a column name <object_id> ' +
                   'to connect static with dynamic data. ' +
                   'Template of this file can be seen in the package documentation')
        
        # display selected data file
        self.selected_static_filename = ctk.CTkLabel(self.tabview.tab("1. Load data"),
                                                      text="No file was selected")
        self.selected_static_filename.grid(row=2, column=2, padx = 10, pady=(5,5), sticky="w")
        
        # display visualize static data
        self.show_data = ctk.CTkButton(self.tabview.tab("1. Load data"), 
                                              anchor='e', 
                                              text="Display/visualize static data", 
                                              command=self.display_static_data)
        self.show_data.grid(row=3, column=2, padx = 10, 
                                   pady=(5,5), sticky="w")
        CTkToolTip(self.show_data, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to have a look at the data (using pandastable)' +
                   'here you can also visualize the data with differnt type of plots')
        
        # check to load/unload
        self.check_static_data = ctk.IntVar(value=0)        
        self.checkbox_static_data = ctk.CTkCheckBox(self.tabview.tab("1. Load data"), 
                                               text="Check/uncheck to use/discard data",
                                               command=self.load_static_data, 
                                               variable=self.check_static_data)
        
        self.checkbox_static_data.grid(row=4, column=2, padx = 10, pady=(5,5), sticky="w")
        CTkToolTip(self.checkbox_static_data, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'If you check this box, data from the selected \n' +
                       'static data file will be used, uncheck means data will \n' +
                       'not be used')
        #--------------------------------------------Content of filter data tab
        # Select input features
        self.select_input_feature_label = ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                                       text="1. Select input features")
        self.select_input_feature_label.grid(row=0, column=0, padx = 10, pady=(5,5), sticky="w")
        self.select_input_feature = CTkListbox(master=self.tabview.tab("2. Filter data"), 
                                               multiple_selection=True, border_width=1.5,
                                               text_color="black")
        self.select_input_feature.grid(row=1, column=0, padx = 10, pady=(5,5), sticky="w")    
        CTkToolTip(self.select_input_feature_label, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Select input features for the neural network. \n' +
                       'You can select as many input features as you want')
        
        # Select target features
        self.select_target_feature_label = ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                                       text="3. Select target features")
        self.select_target_feature_label.grid(row=0, column=2, padx = 10, pady=(5,5), sticky="w")
        CTkToolTip(self.select_target_feature_label, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Select target features (model outputs). \n' +
                       'You can select as many target features as you want')
        self.select_target_feature = CTkListbox(master=self.tabview.tab("2. Filter data"), 
                                                multiple_selection=True, border_width=1.5,
                                               text_color="black")
        self.select_target_feature.grid(row=1, column=2, padx = 10, pady=(5,5), sticky="w") 
        
        # Select object id
        self.object_id_label = ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                                   text="2. Select object id")
        self.object_id_label.grid(row=2, column=0, padx = 10, pady=(5,5), sticky="w")
        CTkToolTip(self.object_id_label, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Select object id (basin id) that you want to put into the model. \n' +
                       'You can select as many object ids as you want')
        self.object_id = CTkListbox(master=self.tabview.tab("2. Filter data"), 
                                           multiple_selection=True, border_width=1.5,
                                           text_color="black")
        self.object_id.grid(row=3, column=0, rowspan=7, padx = 10, pady=(5,5), sticky="w") 
        
        # Trainning period
        self.select_date_train= ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                             text="4. Training period (yyyy-mm-dd)")
        self.select_date_train.grid(row=2, column=2, padx = 10, pady=(5,5), sticky="w")    
        self.start_train = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                         date_pattern= 'yyyy-mm-dd', width = 25,
                                         year=1800, month=1, day=1, font=ctk.CTkFont(size=14))
        self.start_train.grid(row= 3,column=2, padx=30, pady=10, sticky='e')
        self.end_train = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                       date_pattern= 'yyyy-mm-dd', width = 25,
                                       year=2010, month=1, day=1, font=ctk.CTkFont(size=14))
        self.end_train.grid(row= 4,column=2, padx=30, pady=10, sticky='e')   
        CTkToolTip(self.select_date_train, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w', 
                   message='Select starting date (upper calender box) and \n' + 
                   'ending date (lower calendar box) of the training period. \n'+
                   'Training data are used to fit model parameters.')

        # start validation calander  
        self.select_date_valid= ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                            text="5. Validation period (yyyy-mm-dd)")
        self.select_date_valid.grid(row=5, column=2, padx = 10, pady=(5,5), sticky="w")    
        self.start_valid = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                         date_pattern= 'yyyy-mm-dd', width = 25,
                                         year=1800, month=1, day=1, font=ctk.CTkFont(size=14))
        CTkToolTip(self.select_date_valid, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w', 
                   message='Select starting date (upper calender box) and \n' + 
                   'ending date (lower calendar box) of the validtion period. \n'+
                   'Validation data ared used for early stopping.')
        self.start_valid.grid(row= 6,column=2, padx=30, pady=10, sticky='e')
        self.end_valid = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                       date_pattern= 'yyyy-mm-dd', width = 25,
                                       year=2018, month=1, day=1, font=ctk.CTkFont(size=14))
        self.end_valid.grid(row= 7,column=2, padx=30, pady=10, sticky='e')   
        
        # start testing calander  
        self.select_date_test= ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                            text="6. Testing period (yyyy-mm-dd)")
        self.select_date_test.grid(row=8, column=2, padx = 10, pady=(5,5), sticky="w")    
        self.start_test = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                         date_pattern= 'yyyy-mm-dd', width = 25,
                                         year=1800, month=1, day=1, font=ctk.CTkFont(size=14))
        CTkToolTip(self.select_date_test, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w', 
                   message='Select starting date (upper calender box) and \n' + 
                   'ending date (lower calendar box) of the testing period. \n' +
                   'Testing data are used for final evaluation of the model.')
        self.start_test.grid(row= 9,column=2, padx=30, pady=10, sticky='e')
        self.end_test = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                       date_pattern= 'yyyy-mm-dd', width = 25,
                                       year=2018, month=1, day=1, font=ctk.CTkFont(size=14))
        self.end_test.grid(row= 10,column=2, padx=30, pady=10, sticky='e')   
               
        #-----------------------------------------------------3. Transform data
        self.transform_dd_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="1. Transformer for dynamic features")
        self.transform_dd_label.grid(row=0, column=0, pady = 3, padx = 10, sticky = "w")
        
        self.transform_dd_option = ctk.CTkOptionMenu(self.tabview.tab("3. Transform data"),
                                                   values=['MinMaxScaler', 'Z-score', 'None'],
                                                   command=self.transform_dynamic_data_option)
        self.transform_dd_option.grid(row=1, column=0, pady = 3, padx = 10, sticky = "w")
        CTkToolTip(self.transform_dd_option, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to select method for transforming dynamic data') 


        self.transform_tar_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="2. Transformer for target features")
        self.transform_tar_label.grid(row=4, column=0, pady = 3, padx = 10, sticky = "w")
        
        self.transform_tar_option = ctk.CTkOptionMenu(self.tabview.tab("3. Transform data"),
                                                      values=['MinMaxScaler', 'Z-score', 'None'],
                                                      command=self.transform_target_data_option) 
        self.transform_tar_option.grid(row=5, column=0, pady = (3,35), padx = 10, sticky = "w")
        CTkToolTip(self.transform_tar_option, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to select method for transforming target features') 
        
        self.transform_ss_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                       text="3. Transformer for static features")
        self.transform_ss_label.grid(row=2, column=0, pady = 3, padx = 10, sticky = "w")

        self.transform_ss_option = ctk.CTkOptionMenu(self.tabview.tab("3. Transform data"),
                                                 values=['MinMaxScaler', 'Z-score', 'None'],
                                                 command=self.transform_static_data_option) 
        self.transform_ss_option.grid(row=3, column=0, pady = 3, padx = 10, sticky = "w")
        CTkToolTip(self.transform_ss_option, delay=0.1, bg_color = 'orange',
           text_color = 'black', anchor = 'w', 
           message= 'Click here to select method for transforming static data') 


        self.execute_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="4. Data split and transform")
        self.execute_label.grid(row=0, column=2, pady = 3, padx = 10, sticky = "w")
        
        self.execute = ctk.CTkButton(self.tabview.tab("3. Transform data"),
                                                   text="Execute",
                                                   command=self.data_split_transform) 
        self.execute.grid(row=1, column=2, pady = 3, padx = 10, sticky = "w")
        CTkToolTip(self.execute, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to split data (train, valid, test) and transform') 
        
        self.show_result_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="5. Display/Visualize transform data")
        self.show_result_label.grid(row=2, column=2, pady = 3, padx = 10, sticky = "w")
        
        self.show_result = ctk.CTkButton(self.tabview.tab("3. Transform data"),
                                                   text="Display/Visualize",
                                                   command=self.display_orig_trans_data) 
        self.show_result.grid(row=3, column=2, pady = 3, padx = 10, sticky = "w")
        CTkToolTip(self.show_result, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to view the original and transform data of the first object id') 

    #-----------------------------------------------------functions for widgets
    # get dynamic data file name
    def get_dynamic_file(self):
        # No file was selected when just click this button without selecting file
        self.globalData['dynamic_data_file'] = []
        self.globalData['dynamic_data'] = []
        self.globalData['dynamic_data_header'] = []
        self.globalData['object_id'] = []
        
        try:
            # get file name
            file_name = ctk.filedialog.askopenfilename(title="Select dynamic/time series data file", 
                                                       filetypes=(('csv files', '*.csv'),
                                                                  ('All files', '*.*')))
            self.globalData['dynamic_data_file'] = [file_name]
            
            # get data
            self.globalData['dynamic_data'] = pd.read_csv(file_name,
                                                          delimiter=",", header=0)
            
            # get data
            self.globalData['dynamic_data_header'] = list(self.globalData['dynamic_data'].columns)
            self.globalData['object_id'] = list(pd.unique(self.globalData['dynamic_data']["object_id"]))

            # update label
            self.selected_dynamic_filename.configure(text= '...' + file_name[-30:])
            
        except:
            None
            # update label
            self.selected_dynamic_filename.configure(text= 'No file was selected')
        
        # Show message box
        #tk.messagebox.showinfo(title="Done reading data", message="Close this box")   
      
    # show dynamic data in pandas data table
    def display_dynamic_data(self):
        try:
            self.table = Table(tk.Toplevel(self), dataframe=self.globalData['dynamic_data'], 
                               showtoolbar=True, showstatusbar=True)
            self.table.show()
        except:
            None

    # get static data file
    def get_static_file(self):
        
        self.globalData['static_data_file'] = []
        self.globalData['static_data'] = []
        self.globalData['static_data_header'] = []
        
        try:
            file_name = ctk.filedialog.askopenfilename(title="Select static data file", 
                                                  filetypes=(('csv files', '*.csv'),
                                                             ('All files', '*.*')))
            self.globalData['static_data_file'] = [file_name]
            self.globalData['static_data'] = pd.read_csv(file_name,
                                                          delimiter=",", header=0)
            self.globalData['static_data_header'] = list(self.globalData['static_data'].columns)
            
            # update label
            self.selected_static_filename.configure(text= '...' + file_name[-30:])
        except:
            None
            
            # update label
            self.selected_static_filename.configure(text= 'No file was selected')
            
    # show static data in pandas data table
    def display_static_data(self):
        try:
            self.table = Table(tk.Toplevel(self), dataframe=self.globalData['static_data'], 
                               showtoolbar=True, showstatusbar=True)
            self.table.show()  
        except:
            None    

    # Get the data
    def load_dynamic_data(self): 

        try:
            self.select_input_feature.delete(index=0, last='END')
            self.select_target_feature.delete(index=0, last='END')
            self.object_id.delete(index=0, last='END')
        except:
            None
            
        # Get the static data file name
        if self.check_dynamic_data.get() == 0:
            try: 
                del self.config['dynamic_data_file']
            except:
                None
            
        else:
            self.config['dynamic_data_file'] = self.globalData['dynamic_data_file']
            
        try:
            if (("dynamic_data_file" in self.config) & 
                (self.check_dynamic_data.get() == 1)):
                items = self.globalData['dynamic_data_header'].copy()
                object_id = self.globalData['object_id'].copy()
                 
                if (("static_data_file" in self.config) & 
                    (self.check_static_data.get() == 1)):
                    items.extend(self.globalData['static_data_header'])

                # remove columns with object_id and time
                items = [i for i in items if i != "object_id"]
                items = [i for i in items if i != "time"]
                
                for i in items:
                    self.select_input_feature.insert('END', option=i)
                    self.select_target_feature.insert('END', option=i)
                    
                for i in object_id:
                    self.object_id.insert('END', option=i)
            else:
                if (("static_data_file" in self.config) & 
                    (self.check_static_data.get() == 1)):
                    items = self.globalData['static_data_header'].copy()
                     
                    # remove columns with object_id and time
                    items = [i for i in items if i != "object_id"]
                    items = [i for i in items if i != "time"]
                    
                    for i in items:
                        self.select_input_feature.insert('END', option=i)
                        self.select_target_feature.insert('END', option=i)
        except:
            None

           
    def load_static_data(self): 
        
        # First remove content of the select input and target features box
        try:
            self.select_input_feature.delete(index=0, last='END')
            self.select_target_feature.delete(index=0, last='END')

        except:
            None
        
        # Get the static data file name
        if self.check_static_data.get() == 0:
            try: 
                del self.config['static_data_file']
            except:
                None
        else:
            self.config['static_data_file'] = self.globalData['static_data_file']
            
        # Get content for select input and target feature box
        try:
            if (("static_data_file" in self.config) & 
                (self.check_static_data.get() == 1)):
                items = self.globalData['static_data_header'].copy()

                if (("dynamic_data_file" in self.config) & 
                    (self.check_dynamic_data.get() == 1)):
                    items.extend(self.globalData['dynamic_data_header'])

                # remove columns with object_id and time
                items = [i for i in items if i != "object_id"]
                items = [i for i in items if i != "time"]
                
                for i in items:
                    self.select_input_feature.insert('END', option=i)
                    self.select_target_feature.insert('END', option=i)
            else:
                if (("dynamic_data_file" in self.config) & 
                    (self.check_dynamic_data.get() == 1)):
                    items = self.globalData['dynamic_data_header'].copy()

                    # remove columns with object_id and time
                    items = [i for i in items if i != "object_id"]
                    items = [i for i in items if i != "time"]
                    
                    for i in items:
                        self.select_input_feature.insert('END', option=i)
                        self.select_target_feature.insert('END', option=i)

        except:
            None
 
    # Get transformer 
    def transform_dynamic_data_option(self, method: str):
        self.config["scaler_input_dynamic_features"] = [method]
    
    def transform_static_data_option(self, method: str):
        self.config["scaler_input_static_features"] = [method]
        
    def transform_target_data_option(self, method: str):
        self.config["scaler_target_features"] = [method]
        
    def data_split_transform(self):
        # Train test split
        # get training and testing periods
        
        try:
            self.config['train_period'] = pd.to_datetime(
                [self.start_train.get_date(), self.end_train.get_date()],
                format = "%Y-%m-%d %H:%M")

            self.config['valid_period'] = pd.to_datetime(
                [self.start_valid.get_date(), self.end_valid.get_date()],
                format = "%Y-%m-%d %H:%M")
            
            self.config['test_period'] = pd.to_datetime(
                [self.start_test.get_date(), self.end_test.get_date()],
                format = "%Y-%m-%d %H:%M")
        except:
            pass
        
        # get input features
        all_items = self.select_input_feature.get(index='all')
        select_index = self.select_input_feature.curselection()
        input_features = [all_items[i] for i in select_index]
        
        # convert target to static and dynamic features
        if "static_data_file" in self.config:
            self.config['input_static_features'] = []
            self.config['input_dynamic_features'] = []
            
            for feature in input_features: 
                if len(self.globalData['static_data_header']) > 0:
                    if feature in self.globalData['static_data_header']:
                        self.config['input_static_features'].append(feature)
                        
                if len(self.globalData['dynamic_data_header']) > 0:
                    if feature in self.globalData['dynamic_data_header']:
                        self.config['input_dynamic_features'].append(feature)
        else:
            self.config['input_dynamic_features'] = input_features
            try: 
                del self.config['input_static_features']
            except:
                pass
        
        # get target features
        all_items = self.select_target_feature.get(index='all')
        select_index = self.select_target_feature.curselection()
        self.config['target_features'] = [all_items[i] for i in select_index]
            
        # get object id
        all_items = self.object_id.get(index='all')
        select_index = self.object_id.curselection()
        self.config['object_id'] = [all_items[i] for i in select_index]
        
        # Get train and test data
        data = read_train_valid_test_data(self.config)
        
        # Update global data
        self.globalData.update(data)
        
        # Delete data variable to free memory
        del data
         
        # Show message box
        print("Done data split (train, valid, test)")
        
        #--------------------------------------------------------Transform data
        # Get scaler name
        self.globalData["x_scaler_method"], \
            self.globalData["y_scaler_method"] = get_scaler_name(self.config)
        print("done get scaler")
        
        # Scale x_train and x_test
        self.globalData["x_scaler"] = Scaler()
        self.globalData["x_scaler"].fit(x=self.globalData["x_train"], 
                                        method=self.globalData["x_scaler_method"])
        
        self.globalData["x_train_scale"] =\
            self.globalData["x_scaler"].transform(x=self.globalData["x_train"])
        print("done x train scale ")    
            
        self.globalData["x_valid_scale"] =\
            self.globalData["x_scaler"].transform(x=self.globalData["x_valid"]) 
        print("done x valid scale")
            
        self.globalData["x_test_scale"] =\
            self.globalData["x_scaler"].transform(x=self.globalData["x_test"])        
        
        # Scale y_train and y_test
        self.globalData["y_scaler"] = Scaler()
        self.globalData["y_scaler"].fit(x=self.globalData["y_train"], 
                                        method=self.globalData["y_scaler_method"])
        self.globalData["y_train_scale"] =\
            self.globalData["y_scaler"].transform(x=self.globalData["y_train"])
        print("done y scale")
        self.globalData["y_valid_scale"] =\
            self.globalData["y_scaler"].transform(x=self.globalData["y_valid"])   
        print("done y valid scale")
        # Show message box
        tk.messagebox.showinfo(title="Message box", 
                               message="Done data split + transform")
        
    def get_first_tensor(self): 
        
        x_train=self.globalData["x_train"].copy()
        x_train_scale=self.globalData["x_train_scale"].copy()
        
        
        if "input_static_features" in self.config: 
            col_names = self.config["input_dynamic_features"] +\
                self.config["input_static_features"]
        else:
            col_names = self.config["input_dynamic_features"]        
        
        x_train = pd.DataFrame(x_train[next(iter(x_train))])
        x_train.columns = [name + "_origignal" for name in col_names]

        x_train_scale = pd.DataFrame(x_train_scale[next(iter(x_train_scale))])
        x_train_scale.columns = [name + "_transform" for name in col_names]
              
        x_train_scale_cat = pd.concat([x_train, x_train_scale], axis=1)
        
        return x_train_scale_cat
        
    def display_orig_trans_data(self):
        try:
            self.table = Table(tk.Toplevel(self), dataframe=self.get_first_tensor(), 
                               showtoolbar=True, showstatusbar=True)
            self.table.show()
        except:
            pass        
