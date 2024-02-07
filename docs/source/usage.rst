Usage
=====

.. _installation:

Installation
------------

To use HydroEcoLSTM contains many dependencies. To advoild conflict with other
packages, create a virtual environment to install HydroEcoLSMT and its dependencies

To do that, we could install Annaconda from here (https://www.anaconda.com/). Then
open the cmd command and create a virtual environment

.. code-block:: console

>>> # create virtual environment called "saqc-env"
>>> conda create -n hydroecolstm_env

>>> # activate the virtual environment
>>> conda activate hydroecolstm_env

>>> install using pip command
>>> pip install git+https://github.com/tamnva/hydroecolstm.git

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

