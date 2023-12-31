import matplotlib.pyplot as plt

def plot(data: dict, object_id:str,
         train_test_period:str, target_feature:str):  
    
    # Get key of observed and simulated target features
    y_observed = "y_" + train_test_period
    y_simulated = "y_" + train_test_period + "_simulated"
    time = "time_" + train_test_period
    
    # Get index of the target feature
    index = data["y_column_name"].index(target_feature)
    
    # Extract obeserved and target features from data
    y_observed = data[y_observed][object_id][:, index].detach().numpy()
    y_simulated = data[y_simulated][object_id][:, index].detach().numpy()
    time = data[time][object_id]
    
    # Now plot simulated and observed
    plt.plot(time, y_simulated, color = 'blue', label = "Simulated", alpha=0.9, linewidth=0.75)
    plt.plot(time, y_observed, color = 'red', label = "Observed", alpha=0.9, linewidth=0.75)
    plt.title(label=f"Object id = {object_id}, period = {train_test_period}")
    plt.ylabel(target_feature)
    plt.legend()

    return plt
