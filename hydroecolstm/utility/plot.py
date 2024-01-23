import matplotlib.pyplot as plt

def plot(data: dict, object_id:str, target_feature:str):  
    # Get index of the target feature
    index = data["y_column_name"].index(target_feature)
    
    # Extract obeserved and target features from data
    y_observed_valid = data["y_valid"][object_id][:, index].detach().numpy()
    y_observed_test = data["y_test"][object_id][:, index].detach().numpy()
    
    
    y_valid_simulated = data['y_valid_simulated'][object_id][:, index].detach().numpy()
    y_test_simulated = data['y_test_simulated'][object_id][:, index].detach().numpy()
    
    time_valid = data["time_valid"][object_id]
    time_test = data["time_test"][object_id]
    
    
    # Now plot simulated and observed

    plt.plot(time_valid[5:], y_observed_valid[5:], color = 'red', label = "Observed (valid period)", alpha=0.1, linewidth=1)
    plt.plot(time_test[5:], y_observed_test[5:], color = 'red', label = "Observed (test period)", alpha=0.9, linewidth=1)   
    plt.plot(time_valid[5:], y_valid_simulated[5:], color = 'blue', label = "Simulated (valid period)", alpha=0.1, linewidth=1)
    plt.plot(time_test[5:], y_test_simulated[5:], color = 'blue', label = "Simulated (test period)", alpha=0.9, linewidth=1)
    plt.title(label=f"Catchment = {object_id}")
    plt.ylabel(target_feature)
    plt.legend()

    return plt

def plot_train_valid_loss(loss):
    #loss = data["loss"]
    plt.rcParams.update({'font.size': 8})
    
    epoch = loss.query('best_model == True').iloc[-1,:].epoch
    plt.plot(loss["epoch"], loss["train_loss"], label="Training loss")
    plt.plot(loss["epoch"], loss["validation_loss"], label="Validation loss")
    plt.axvline(x=epoch, label="Best model at epoch " + str(epoch), 
                color = "grey", linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    return plt