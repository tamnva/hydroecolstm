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
The input data format must be in comma seperated values (.csv) file format. There
are two types of input data:

* dynamic (time series) input data: These data are required (a MUST). A template of this file (e.g., can be found `here <https://github.com/tamnva/hydroecolstm/blob/master/examples/1_streamflow_simulation/data/time_series.csv>`_.

* static (catchment attributes) data: These are optional data. An example of such input data file is `here <hhttps://github.com/tamnva/hydroecolstm/blob/master/examples/1_streamflow_simulation/data/static_attributes.csv>`_.

.. note::

   Both dynamic and static data files MUST have a column with a name ``object_id`` which could be the catchment name or id. This ``object_id`` is used to link the two files together, e.g., with a specific ``object_id``, HydroEcoLSTM knows where are the corresponding dynamics and static data.

   The dynamic data file MUST have a column ``time`` in format ``YYYY-MM-DD HH:MM`` (for example, ``2024-12-13 11:30``)

   Inputs and target outputs MUST be in the dynamic data file.

The dynamic data file
################################################################################

Streamflow and instream isotope signatures
-----------------------------------------
TODO: Write something here

The Graphical User Interface
===========================

CAMEL-CH data
-------------
The subset of CAMEL-CH data are in

Streamflow and instream isotope signatures
------------------------------------------
Write something here

The graphical user interface can be called using the following command

.. code-block:: python

   import hydroecolstm
   hydroecolstm.interface.show_gui()


Illustrative examples
=====================

Streamflow prediction in ungauged catchments
--------------------------------------------
Write something here

Multiple target variable predictions
-------------------------------------
Write something here