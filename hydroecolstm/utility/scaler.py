import torch

# scaler
class Scaler:
    def fit(self, x=None, method=None):
        # concatenat all object_id
        for i, object_id in zip(range(len(x)), x):
            if i == 0:
                cat_x = x[object_id]
            else:
                cat_x = torch.cat((cat_x, x[object_id]),0)
        
        # Get either min max or mean and standard deviation 
        self.mins = _column_mins(cat_x)
        self.maxs = _column_maxs(cat_x)
        
        # Get means and standar deviation
        self.means = _column_means(cat_x)
        self.stds = _column_stds(cat_x)
        
        scaler_a = []
        scaler_b = []
        
        for i, method_name in zip(range(len(method)), method):
            if method_name == "MinMaxScaler":
                scaler_a.append(self.mins[i])
                scaler_b.append(self.maxs[i] - self.mins[i])
            elif method_name=="Z-score":
                scaler_a.append(self.means[i])
                scaler_b.append(self.stds[i])
            elif method_name=="None":
                scaler_a.append(0.0)
                scaler_b.append(1.0)
            else:
                print("Error: unknown scaler")
                SystemExit("Program stop, please change scaler")
        
        scaler_ab = torch.cat((torch.tensor(scaler_a, dtype=torch.float32),
                               torch.tensor(scaler_b, dtype=torch.float32)), 0)
        
        self.scaler_parameter = torch.reshape(scaler_ab, 
                                              (2,len(scaler_a)))
 
    def transform(self, x:dict[str:torch.tensor]=None) -> list: 
        x_scale = {}
        for object_id in x:
            x_scale[object_id] =  torch.div(torch.sub(x[object_id], 
                                                      self.scaler_parameter[0,:]), 
                                            self.scaler_parameter[1,:])               
        return x_scale

    def inverse(self, x:list=None) -> list:        
        x_inverse = {}
        for object_id in x:
            x_inverse[object_id] =  torch.add(self.scaler_parameter[0,:],
                                              x[object_id]*self.scaler_parameter[1,:])
        return x_inverse

def _column_mins(input_tensor: torch.tensor=None):
    mask = ~torch.isnan(input_tensor) 
    colMins = [torch.min(torch.masked_select(input_tensor[:,i], mask[:,i]))
               for i in range(input_tensor.shape[1])]
    return torch.tensor(colMins, dtype=torch.float32)

def _column_maxs(input_tensor: torch.tensor=None):
    mask = ~torch.isnan(input_tensor) 
    colMaxs = [torch.max(torch.masked_select(input_tensor[:,i], mask[:,i]))
               for i in range(input_tensor.shape[1])]
    return torch.tensor(colMaxs, dtype=torch.float32)

def _column_means(input_tensor: torch.tensor=None):
    mask = ~torch.isnan(input_tensor) 
    colMeans = [torch.mean(torch.masked_select(input_tensor[:,i], mask[:,i]))
                for i in range(input_tensor.shape[1])]   
    return torch.tensor(colMeans, dtype=torch.float32)

def _column_stds(input_tensor: torch.tensor=None):
    mask = ~torch.isnan(input_tensor) 
    col_means = _column_means(input_tensor)
    
    col_stds = []
    
    for i in range(input_tensor.shape[1]):
        column = torch.masked_select(input_tensor[:,i], mask[:,i])
        col_stds.append((sum((column - col_means[i])**2)/len(column))**0.5)
    return torch.tensor(col_stds, dtype=torch.float32)

# Get scaler name in list
def get_scaler_name(config):
    
    if "input_static_features" not in config.keys():
        no_static_features = 0
    else:
        no_static_features = len(config["input_static_features"])
        
        
    # Get name of scaler for dynamic input
    scaler_name_input = config["scaler_input_dynamic_features"]*\
        len(config["input_dynamic_features"])
    
    # replicate this n times
    if no_static_features > 0 and\
        "scaler_input_static_features" in config:
            for name in config["scaler_input_static_features"]*no_static_features:
                scaler_name_input.append(name)
        
    # scaler name target
    scaler_name_target = config["scaler_target_features"]*len(config["target_features"])
    
    return scaler_name_input, scaler_name_target

