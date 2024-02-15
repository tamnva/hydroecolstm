Quick start
===========
.. quick_start:

Installation
------------

To use HydroEcoLSTM contains many dependencies. To avoid conflict with other
packages, create a virtual environment to install HydroEcoLSMT and its dependencies.
To do that, we could install Anaconda from here (https://www.anaconda.com/). Then
open the cmd command to:

* create a virtual environment,
* activate the environment,
* install HydroEcoLSTM using pip command.

.. code-block::

    conda create -n hydroecolstm_env
    conda activate hydroecolstm_env
    pip install git+https://github.com/tamnva/hydroecolstm.git

Example data
------------
Example data for streamflow simulation are part of the CAMLE-CH data and


The graphical user interface
----------------------------

The graphical user interface can be called using the following command

.. code-block:: python

   import hydroecolstm
   hydroecolstm.interface.show_gui()


The graphical user interface
----------------------------