Configuration file config.yml
=============================

* ``dynamic_data_file``: path to the dynamic (time series data) file.

* ``static_data_file``: [``list``] - path to the static (catchment attributes) data file (optional input).

* ``output_directory``: [``list``] - path to the output folder (where files created by HydroEcoLSTM will be saved).

* ``input_static_features``: [``list``] - input statics (catchment attributes) features, the list of input statics features are the column names (except the ``object_id`` and ``time`` columns) of the statics data file.

* ``input_dynamic_features``: [``list``] - input dynamic (catchment attributes) features, the list of input dynamic features are the column names (execept the ``object_id`` column)  of the dynamic data file.

* ``target_features``: [``list``] - target features, the list of target features are the column names (execept the ``object_id`` column and the names that are alreay in `input_dynamic_features``) of the dynamic data file.

* ``object_id``: [``list``] - the catchment ID used to train the model, it can be any object_id listed in the ``dynamic_data_file`` and ``static_data_file`` files.

* ``train_period``: [``list``] - the starting and ending time of the trainning period, muss be in ``YYYY-MM-DD HH:MM`` format.

* ``valid_period``: [``list``] - the starting and ending time of the validation period, muss be in ``YYYY-MM-DD HH:MM`` format.

* ``test_period``: [``list``] - the starting and ending time of the test period, muss be in ``YYYY-MM-DD HH:MM`` format.

* ``model_class: [``str``] - name of the LSTM models, could be ``LSTM`` or ``EA-LSTM``

* ``Regression``: [``list``] - configuratoin of the model head, containing the following keys (which is also a list):

* ``activation_function``: [``list``] - name of the activation function for each layer, could be a list of character ``Identity``, ``ReLu``, ``Sigmoid``, ``Tanh``, ``Softplus``.

* ``num_neurons``: [``list``] - number of neurons in each layers of the model head, use ``None`` for the last layer as the number of neurons in this layer is defined by the model, which is equals to the number of target features

* ``num_layers``: [``int``] - number of layers of the model head.

* ``scaler_input_dynamic_features``: [``list``] - name of the transformation technique for the input dynamic features, for example ``Z-score``, ``MinMaxScaler``, or ``None``
 
* ``scaler_input_static_features``: [``list``] - name of the transformation technique for the input static features, for example ``Z-score``, ``MinMaxScaler``, or ``None`

* ``scaler_target_features``: [``list``] - name of the transformation technique for the target features, for example ``Z-score``, ``MinMaxScaler``, or ``None`

* ``hidden_size``: [``int``] - hidden size of the LSTM.

* ``num_layers``: [``int``] - number of layers of the LSTM.

* ``n_epochs``: [``int``] - number of training epochs.

* ``learning_rate``: [``float``] - learning rate.

* ``dropout``: [``float``] - dropout rate, applied for the output of each LSTM layer (even there is only a single LSMT layer).

* ``warmup_length``: [``int``] - numer of warmup time steps, must be less than the ``sequence_length``. For example, if the ``sequence_length = 100`` and the ``warmup_length = 10``, only the last 90 values of the target features are used when calculating loss.

* ``loss_function``: [``str``] - name of the loss function used for model training, could be the root mean squared error ``RMSE``, mean absolute error ``MAE``, or mean squared error``MSE``.

* ``sequence_length``: [``int``] - sequence length.

* ``batch_size``: [``int``] - batch size.

* ``patience``: [``int``] - number of epoch to wait to see if there is no improvements in the tranning loss then stop the traning, more detail, please see the description from `Bjarte Mehus Sunde  <https://github.com/Bjarten/early-stopping-pytorch>`_ .

* ``eval_function``: [``int``] - name of the function for calculate model performance, ``MSE``, ``RMSE``, Nash-SutCliffe efficiency ``NSE``, ``MAE`` (this is not used during model trainning), just in case you want to calculate some of the model performance statistics to shown in the report .

* ``static_data_file_forecast``: [``list``] - path to the static (catchment attributes) data file that contain data of the ungauged catchments or of the forecast period, which I call forecast in general. If it is the same file as ``static_data_file`` then type ``static_data_file``.
  - static_data_file

* ``dynamic_data_file_forecast``: [``list``] - path to the dynamic (time series) data file that contain data of the ungauged catchments or of the forecast period. If it is the same file as ``dynamic_data_file`` then type ``dynamic_data_file``.

* ``forecast_period``: [``list``] - the starting and ending time of the forecast period, muss be in ``YYYY-MM-DD HH:MM`` format.

* ``object_id_forecast``: [``list``] - list of the object_id in the ``static_data_file_forecast`` that you want to used

The configuration file will be read as a ``dict`` type object, so you can also create this configuration file in Python as a list object. 
