Installation
===========
.. Installation with Anaconda

HydroEcoLSTM was developed based on many other packages. To avoid conflict with other pre-installed packages, create a virtual environment to install HydroEcoLSMT and its dependencies. To do that, we could install Anaconda from `here <https://www.anaconda.com/>`_. Download the`environment file <https://github.com/tamnva/hydroecolstm/tree/master/environments/>`_ Then open the Anaconda PowerShell Prompt:

.. code-block:: console
    
    # 1. Create the environment (from the downloaded file environment.yml)
    conda env create -f environment.yml
    conda activate hydroecolstm_env

    # Install the latest HydroEcoLSTM from github
    pip install git+https://github.com/tamnva/hydroecolstm.git
	
