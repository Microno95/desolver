
DESolver
========


.. image:: https://travis-ci.com/Microno95/desolver.svg?branch=master
   :target: https://travis-ci.com/Microno95/desolver
   :alt: Build Status

.. image:: https://readthedocs.org/projects/desolver/badge/?version=latest
    :target: https://desolver.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/Microno95/desolver/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/Microno95/desolver
   :alt: codecov

.. image:: https://bettercodehub.com/edge/badge/Microno95/desolver?branch=master
   :target: https://bettercodehub.com/
   :alt: BCH compliance


This is a python package for solving Initial Value Problems using various numerical integrators.
Many integration routines are included ranging from fixed step to symplectic to adaptive integrators.

Documentation
=============

Documentation is now available at `desolver docs <https://desolver.readthedocs.io/>`_! This will be updated with new examples as they are written, currently the examples show the use of ``pyaudi``.

Latest Release
==============

**3.0.0** - PyAudi support has been finalised. It is now possible to do numerical integrations using ``gdual`` variables such as ``gdual_double``\ , ``gdual_vdouble`` and ``gdual_real128`` (only on select platforms, refer to `pyaudi docs <https://darioizzo.github.io/audi/>`_ for more information). Install desolver with pyaudi support using ``pip install desolver[pyaudi]``. Documentation has also been added and is available at `desolver docs <https://desolver.readthedocs.io/>`_.

**2.5.0** - Event detection has been added to the module. It is now possible to do numerical integration with terminal and non-terminal events.

**2.2.0** - PyTorch backend is now implemented. It is now possible to numerically integrate a system of equations that use pytorch tensors and then compute gradients from these.

Use of PyTorch backend requires installation of PyTorch from `here <https://pytorch.org/get-started/locally/>`_.

To Install:
===========

Just type

``pip install desolver``

Implemented Integration Methods
-------------------------------

Explicit Methods
~~~~~~~~~~~~~~~~

Adaptive Methods
^^^^^^^^^^^^^^^^

#. Runge-Kutta 14(12) with Feagin Coefficients [\ **NEW**\ ]
#. Runge-Kutta 10(8) with Feagin Coefficients [\ **NEW**\ ]
#. Runge-Kutta 8(7) with Dormand-Prince Coefficients [\ **NEW**\ ]
#. Runge-Kutta 4(5) with Cash-Karp Coefficients
#. Adaptive Heun-Euler Method

Fixed Step Methods
^^^^^^^^^^^^^^^^^^

#. Runge-Kutta 4 - The classic RK4 integrator
#. Runge-Kutta 5 - The 5th order integrator from RK45 with Cash-Karp Coefficients.
#. BABs9o7H Method  -- Based on arXiv:1501.04345v2 - BAB's9o7H
#. ABAs5o6HA Method -- Based on arXiv:1501.04345v2 - ABAs5o6H
#. Midpoint Method
#. Heun's Method
#. Euler's Method
#. Euler-Trapezoidal Method

Implicit Methods
~~~~~~~~~~~~~~~~
**NOT YET IMPLEMENTED**

Minimal Working Example
=======================

This example shows the integration of a harmonic oscillator using DESolver.

.. code-block:: python

   import desolver as de
   import desolver.backend as D

   def rhs(t, state, k, m, **kwargs):
       return D.array([[0.0, 1.0], [-k/m,  0.0]])@state

   y_init = D.array([1., 0.])

   a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=1e-9, atol=1e-9, constants=dict(k=1.0, m=1.0))

   print(a)

   a.integrate()

   print(a)

   print("If the integration was successful and correct, a[0].y and a[-1].y should be near identical.")
   print("a[0].y  = {}".format(a[0].y))
   print("a[-1].y = {}".format(a[-1].y))

   print("Maximum difference from initial state after one oscillation cycle: {}".format(D.max(D.abs(a[0].y-a[-1].y))))
