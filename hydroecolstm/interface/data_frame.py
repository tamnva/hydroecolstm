
import customtkinter as ctk
import tkinter as tk
import pandas as pd
from pandastable import Table
import tkcalendar as tkc
from CTkListbox import CTkListbox
from CTkToolTip import CTkToolTip

from hydroecolstm.utility.scaler import Scaler, get_scaler_name
from hydroecolstm.data.read_data import read_train_test_data

class DataFrame(ctk.CTkScrollableFrame):
    def __init__(self, container=None, config=None, globalData=None):
        super().__init__(container)
        
        # setup the grid layout manager
        self.config = config
        self.globalData = globalData
        self.__create_widgets()         
        self.tableframe = tk.Frame()
        
    # create widgets for sidebar frame
    def __create_widgets(self): 
        
        # create tabs
        self.tabview = ctk.CTkTabview(master=self, width = 750, border_width=1.5,
                                      fg_color = "transparent")
        self.tabview.pack(fill='both',expand=1)
        self.tabview.add("1. Load data")
        self.tabview.tab("1. Load data").grid_columnconfigure((0,1), weight=1)
        self.tabview.add("2. Filter data")
        self.tabview.tab("2. Filter data").grid_columnconfigure((0,1), weight=1)
        self.tabview.add("3. Transform data")
        self.tabview.tab("3. Transform data").grid_columnconfigure((0), weight=1)
        
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
                   'The data file must be in .csv format with header, seperator' + 
                   ' is <,>. This data file must contains at least two colums, ' +
                   'one with the name <object_id> and the other column with the ' +
                   'name <time> in format yyyy-mm-dd hh:mm.The object_id column will '+ 
                   'link object in dynamic/timeseries data file with static data file. ' +
                   'Timeseries of BOTH input and target features must be in this file. ' +
                   'Template of this file can be seen in this Github repo ./example_1')
        
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
        self.static_label.grid(row=0, column=1, padx = 10, pady=(5,5), sticky="e")
        
        # select static data file
        self.select_static_file = ctk.CTkButton(self.tabview.tab("1. Load data"), 
                                              anchor='e', 
                                              text="Select static data file", 
                                              command=self.get_static_file)

        self.select_static_file.grid(row=1, column=1, padx = 10, 
                                   pady=(5,5), sticky="e")
        CTkToolTip(self.select_static_file, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w',  wraplength=500, 
                   message='Click here to select static (catchment attributes) data file. ' + 
                   'the data file must be in .csv format with header, seperator ' + 
                   ' is <,>. This data file must contains at a column name ' +
                   '<object_id> to connect static with dynamic data. ' +
                   'Template of this file can be seen in this Github repo ./example_1')
        
        # display selected data file
        self.selected_static_filename = ctk.CTkLabel(self.tabview.tab("1. Load data"),
                                                      text="No file was selected")
        self.selected_static_filename.grid(row=2, column=1, padx = 10, pady=(5,5), sticky="e")
        
        # display visualize static data
        self.show_data = ctk.CTkButton(self.tabview.tab("1. Load data"), 
                                              anchor='e', 
                                              text="Display/visualize static data", 
                                              command=self.display_static_data)
        self.show_data.grid(row=3, column=1, padx = 10, 
                                   pady=(5,5), sticky="e")
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
        
        self.checkbox_static_data.grid(row=4, column=1, padx = 10, pady=(5,5), sticky="e")
        CTkToolTip(self.checkbox_static_data, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'If you check this box, data from the selected \n' +
                       'static data file will be used, uncheck means data will \n' +
                       'not be used')
        #--------------------------------------------content of filter data tab
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
                                                       text="2. Select target features")
        self.select_target_feature_label.grid(row=0, column=1, padx = 10, pady=(5,5), sticky="e")
        CTkToolTip(self.select_target_feature_label, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Select target features (model outputs). \n' +
                       'You can select as many target features as you want')
        self.select_target_feature = CTkListbox(master=self.tabview.tab("2. Filter data"), 
                                                multiple_selection=True, border_width=1.5,
                                               text_color="black")
        self.select_target_feature.grid(row=1, column=1, padx = 10, pady=(5,5), sticky="e") 
        
        # Select object id
        self.object_id_label = ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                                   text="3. Select object id")
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
        self.select_date_train.grid(row=2, column=1, padx = 10, pady=(5,5), sticky="e")    
        self.start_train = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                         date_pattern= 'yyyy-mm-dd', width = 25,
                                         year=1800, month=1, day=1, font=ctk.CTkFont(size=16))
        self.start_train.grid(row= 3,column=1, padx=30, pady=10, sticky='e')
        self.end_train = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                       date_pattern= 'yyyy-mm-dd', width = 25,
                                       year=2010, month=1, day=1, font=ctk.CTkFont(size=16))
        self.end_train.grid(row= 4,column=1, padx=30, pady=10, sticky='e')   
        CTkToolTip(self.select_date_train, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w', 
                   message='Select starting date (upper calender box) and \n' + 
                   'ending date (lower calendar box) of the training period')
        
        # start testing calander  
        self.select_date_test= ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                            text="5. Testing period (yyyy-mm-dd)")
        self.select_date_test.grid(row=5, column=1, padx = 10, pady=(5,5), sticky="e")    
        self.start_test = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                         date_pattern= 'yyyy-mm-dd', width = 25,
                                         year=1800, month=1, day=1, font=ctk.CTkFont(size=16))
        CTkToolTip(self.select_date_test, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor='w', 
                   message='Select starting date (upper calender box) and \n' + 
                   'ending date (lower calendar box) of the testing period')
        self.start_test.grid(row= 6,column=1, padx=30, pady=10, sticky='e')
        self.end_test = tkc.DateEntry(self.tabview.tab("2. Filter data"), 
                                       date_pattern= 'yyyy-mm-dd', width = 25,
                                       year=2018, month=1, day=1, font=ctk.CTkFont(size=16))
        self.end_test.grid(row= 7,column=1, padx=30, pady=10, sticky='e')   
        

        # Start subsetting data
        self.data_filter_label = ctk.CTkLabel(self.tabview.tab("2. Filter data"),
                                              text="6. Start subseting/filtering data")
        self.data_filter_label.grid(row=8, column=1, padx = 10, pady=(5,5), sticky="e")        
        self.data_filter = ctk.CTkButton(self.tabview.tab("2. Filter data"),
                                         anchor='e',
                                         text="Filter and split data",
                                         command=self.read_train_and_test_data)
        self.data_filter.grid(row=9, column=1, padx = 10, pady=(5,5), sticky="en")
        CTkToolTip(self.data_filter, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click this button to start spliting your data into \n' +
                       'training, testing datasets')        
        #-----------------------------------------------------3. Transform data
        self.transform_dd_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="1. Transformer for dynamic features")
        self.transform_dd_label.pack(anchor="w")
        
        self.transform_dd_option = ctk.CTkOptionMenu(self.tabview.tab("3. Transform data"),
                                                   values=['MinMaxScaler', 'Z-score', 'None'],
                                                   command=self.transform_dynamic_data_option)
        self.transform_dd_option.pack(anchor="w")
        CTkToolTip(self.transform_dd_option, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to select which method should be used for \n' + 
                   ' transforming the dynamic data. None means the data will not be transformed \n') 

        self.transform_ss_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="2. Transformer for static features")
        self.transform_ss_label.pack(anchor="w")
        
        self.transform_ss_option = ctk.CTkOptionMenu(self.tabview.tab("3. Transform data"),
                                                  values=['MinMaxScaler', 'Z-score', 'None'],
                                                   command=self.transform_static_data_option) 
        self.transform_ss_option.pack(anchor="w")
        CTkToolTip(self.transform_ss_option, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to select which method should be used for \n' + 
                   ' transforming the static data. None means the data will not be transformed \n') 

        self.transform_tar_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="3. Transformer for target features")
        self.transform_tar_label.pack(anchor="w")
        
        self.transform_tar_option = ctk.CTkOptionMenu(self.tabview.tab("3. Transform data"),
                                                      values=['MinMaxScaler', 'Z-score', 'None'],
                                                      command=self.transform_target_data_option) 
        self.transform_tar_option.pack(anchor="w")
        CTkToolTip(self.transform_tar_option, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to select which method should be used for \n' + 
                   ' transforming the target data. None means the data will not be transformed \n') 
        
        self.execute_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="4. Execute data transformation")
        self.execute_label.pack(anchor="w", pady=(50,5))
        
        self.execute = ctk.CTkButton(self.tabview.tab("3. Transform data"),
                                                   text="Execute",
                                                   command=self.transform_data) 
        self.execute.pack(anchor="w")
        CTkToolTip(self.execute, delay=0.1, bg_color = 'orange',
                   text_color = 'black', anchor = 'w', 
                   message= 'Click here to execute data transformation') 
        
        self.show_result_label = ctk.CTkLabel(self.tabview.tab("3. Transform data"),
                                               text="5. Display/Visualize transform data")
        self.show_result_label.pack(anchor="w")
        
        self.show_result = ctk.CTkButton(self.tabview.tab("3. Transform data"),
                                                   text="Display/Visualize",
                                                   command=self.display_orig_trans_data) 
        self.show_result.pack(anchor="w")
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
   
    # Assemble all the information to config and perform read_train_and_test_data
    def read_train_and_test_data(self):        
        # get training and testing periods
        self.config['train_period'] = [self.start_train.get_date(),
                                       self.end_train.get_date()]
        self.config['test_period'] = [self.start_test.get_date(),
                                      self.end_test.get_date()]
        
        # get input features
        all_items = self.select_input_feature.get(index='all')
        select_index = self.select_input_feature.curselection()
        input_features = [all_items[i] for i in select_index]
        
        # convert target to static and dynamic features
        if "static_data_file" in self.config:
            #print("static in self config")
            #print(self.config)
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
                None   
                
        #print("summayr config")
        #print(self.config)
        
        # get target features
        all_items = self.select_target_feature.get(index='all')
        select_index = self.select_target_feature.curselection()
        self.config['target_features'] = [all_items[i] for i in select_index]
            
        # get object id
        all_items = self.object_id.get(index='all')
        select_index = self.object_id.curselection()
        self.config['object_id'] = [all_items[i] for i in select_index]
        
        # Get train and test data
        data_train_test_split = read_train_test_data(self.config)
        self.globalData["x_train"] = data_train_test_split["x_train"]
        self.globalData["y_train"] = data_train_test_split["y_train"]
        self.globalData["time_train"] = data_train_test_split["time_train"]
        
        self.globalData["x_test"] = data_train_test_split["x_test"]
        self.globalData["y_test"] = data_train_test_split["y_test"]
        self.globalData["time_test"] = data_train_test_split["time_test"]
        
        self.globalData["x_column_name"] = data_train_test_split["x_column_name"]
        self.globalData["y_column_name"] = data_train_test_split["y_column_name"]
        
        del data_train_test_split
         
        # Show message box
        tk.messagebox.showinfo(title="Message box", 
                               message="Done filtering/spliting data")
 

    # Get transformer 
    def transform_dynamic_data_option(self, method: str):
        self.config["scaler_input_dynamic_features"] = [method]
    
    def transform_static_data_option(self, method: str):
        self.config["scaler_input_static_features"] = [method]
        
    def transform_target_data_option(self, method: str):
        self.config["scaler_target_features"] = [method]
        
    def transform_data(self):
        # Get scaler name
        self.globalData["x_scaler_method"], \
            self.globalData["y_scaler_method"] = get_scaler_name(self.config)

        #print(self.globalData["x_scaler_method"])
        #print(self.globalData["y_scaler_method"])
        #print(self.globalData["x_train"].keys())
        # Scale x_train and x_test
        self.globalData["x_scaler"] = Scaler()
        self.globalData["x_scaler"].fit(x=self.globalData["x_train"], 
                                        method=self.globalData["x_scaler_method"])
        self.globalData["x_train_scale"] =\
            self.globalData["x_scaler"].transform(x=self.globalData["x_train"])
        self.globalData["x_test_scale"] =\
            self.globalData["x_scaler"].transform(x=self.globalData["x_test"])        
        
        # Scale y_train and y_test
        self.globalData["y_scaler"] = Scaler()
        self.globalData["y_scaler"].fit(x=self.globalData["y_train"], 
                                        method=self.globalData["y_scaler_method"])
        self.globalData["y_train_scale"] =\
            self.globalData["y_scaler"].transform(x=self.globalData["y_train"])
        self.globalData["y_test_scale"] =\
            self.globalData["y_scaler"].transform(x=self.globalData["y_test"])   
            
        # Show message box
        tk.messagebox.showinfo(title="Message box", 
                               message="Done transforming data")
        
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
            #print(self.get_first_tensor())
        except:
            None        
