Installation
===========
.. Installation:
01234567890123456789012345678901234567890123456789012345678901234567890123456789

To use HydroEcoLSTM contains many dependencies. To avoid conflict with other
packages, create a virtual environment to install HydroEcoLSMT and its dependencies.
To do that, we could install Anaconda from `here <https://www.anaconda.com/>`_.
open the cmd command to:

* create a virtual environment,
* activate the environment,
* install HydroEcoLSTM using pip command.

.. code-block::

    conda create -n hydroecolstm_env
    conda activate hydroecolstm_env
    pip install git+https://github.com/tamnva/hydroecolstm.git



Data
====

Input data format
-----------------
The input data format must be in comma seperated values (.csv) file. There are two types of input data, and they must be in two files:

* Dynamic (time series) input data file: This file is required (a MUST). A template of this file (e.g., can be found `here <https://github.com/tamnva/hydroecolstm/blob/master/examples/1_streamflow_simulation/data/time_series.csv>`_. This file contains time series data for all catchments.

* Static (catchment attributes) input data file: These are optional data. An example of such input data file is `here <https://github.com/tamnva/hydroecolstm/blob/master/examples/1_streamflow_simulation/data/static_attributes.csv>`_. This file contains the attributes of all catchments.

.. note::

   Both dynamic and static data files MUST have a column with a name ``object_id`` which could be the catchment name or id. This ``object_id`` is used to link the two files together, e.g., with a specific ``object_id``, HydroEcoLSTM knows where are the corresponding dynamics and static data.

   The dynamic data file MUST have a column ``time`` in format ``YYYY-MM-DD HH:MM`` (for example, ``2024-12-13 11:30``)

   Inputs and target outputs MUST be in the dynamic data file.
   
Statics input data file is needed when you model, for example, streamflow for multiple catchments. For a single catchment, this data (file) does not required. However, a recent `paper in HESS <https://doi.org/10.5194/hess-2023-275>`_ argues that we should not train LSTM for a single catchment.

Example data
============

CAMEL-CH data
-------------
The `example data  <https://github.com/tamnva/hydroecolstm/blob/master/examples/1_streamflow_simulation/data>`_ are only the subset of the `CAMEL-CH data  <https://doi.org/10.5194/essd-15-5755-2023>`_. This data contains dynamic (``time_series.csv``) and static (``static_attributes.csv``) data of 10 catchments from the CAMEL-CH data. The ``time_series.csv`` contains ``discharge_vol_m3_s``,  ``precipitation_mm_d``, ``temperature_min_degC``,``temperature_mean_degC``, ``temperature_max_degC``, and ``rel_sun_dur``. The input units does not matter for all dynamic input data, for example, you can used different unit for discharge (such as mm/day or cubic feet meter per second). This is becuase LSTMs does not based on the mass balance equations. What important is that the unit MUST be the same for all catchments (which I named the catchment ID as ``object_id``) (e.g., you cannot use the unit of discharge is m3/s for the first catchments and mm/day for the second catchment). Same are applied for the units of catchment characteristics in the ``static_attributes.csv`` file.

Please refer to the `Hoege et al. (2023)  <https://doi.org/10.5194/essd-15-5755-2023>`_ for a detailed description of the CAMEL-CH data.

Stable isotope data
-------------------
The `second dataset <https://github.com/tamnva/hydroecolstm/tree/master/examples/2_streamflow_isotope_simulation/data>`_ are the high frequency isotope data in precipitation and streamflow in the Alp and Erlenbach catchment in Switzerland. 

Please refer to the `von Freyberg et al. (2022) <https://doi.org/10.1038/s41597-022-01148-1>`_ for a detailed description of the data.


The graphical user interface
============================

The graphical user interface can be called using the following command:

.. code-block:: python

   import hydroecolstm
   hydroecolstm.interface.show_gui()
