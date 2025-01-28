.. installation

Installing desolver
===================

`desolver` is a pure python module that builds upon the functionalities provided by numpy for a relatively efficient numerical integration library.

Furthermore, as `desolver` relies on `autoray <https://github.com/jcmgray/autoray>` for the backend of array/tensors operations, `desolver` can easily extend to the use of other libraries such as pytorch, tensorflow, etc. in order to leverage accelerators and optimised gradient computation schemes for faster numerical integration.

With the use of libraries equipped with automatic differentiation, it is possible to compute the gradients of the integrated system.

To install `desolver` simply type:

.. code-block:: bash

   pip install desolver
   
Refer to the following links for `pytorch support <https://pytorch.org/get-started/>`_. `desolver`` will automatically detect these modules and appropriately leverage their functionality.

Unlike versions prior to v5.0.0, `desolver` no longer requires specifying the backend manually and relies on `autoray` to automatically use the appropriate backend. Conversely, this puts the onus on the user to ensure that the functions remain pure `pytorch` or pure `numpy`, or to implement gradients of their function when necessary.

.. code-block:: python
   
   import desolver

To check that all went well fire-up your python console and try the example in :ref:`quick-start example <Quick Start>`.