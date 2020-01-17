.. installation

Installing desolver
===================

`desolver` is a pure python module that builds upon the functionalities provided by numpy for a relatively efficient numerical integration library.

Furthermore, desolver incoporates the use of `pyaudi <https://darioizzo.github.io/audi/>`_ and `pytorch <https://pytorch.org/>`_ for more advanced applications.

With these libraries, it is possible to incorporate gradient descent of parameters into the numerical integration of some system. :ref:`Differential Intelligence` example shows how to use pyaudi to tune the weights of a controller based on the outcome of a numerical integration.

To install `desolver` simply type:

.. code-block:: bash

   pip install desolver
   
Refer to the following links for `pyaudi support <https://darioizzo.github.io/audi/install_c.html>`_ and `pytorch support <https://pytorch.org/get-started/>`_. desolver will automatically detect these modules if they are in the same environment and use them if possible.

It is necessary to tell desolver if you'd like to use numpy (and pyaudi if available) or pytorch as the backend for the computations. To do this, you can set the environment variable `DES_BACKEND` to either `numpy` or `torch` as shown in the snippet below:

.. code-block:: python

   import os
   os.environ['DES_BACKEND'] = 'torch' # or 'numpy'
   
   import desolver

To check that all went well fire-up your python console and try the example in :ref:`quick-start example <Quick Start>`.