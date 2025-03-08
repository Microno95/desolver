
DESolver
========


.. image:: https://github.com/Microno95/desolver/actions/workflows/pytest-ubuntu.yml/badge.svg
   :target: https://github.com/Microno95/desolver/actions/workflows/pytest-ubuntu.yml
   :alt: Build Status

.. image:: https://readthedocs.org/projects/desolver/badge/?version=latest
    :target: https://desolver.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/Microno95/desolver/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/Microno95/desolver
   :alt: codecov


This is a python package for solving Initial Value Problems using various numerical integrators.
Many integration routines are included ranging from fixed step to symplectic to adaptive integrators.

Documentation
=============

Documentation is now available at `desolver docs <https://desolver.readthedocs.io/>`_! This will be updated with new examples as they are written.

To Install:
===========

Just type

``pip install desolver``

Use of PyTorch backend requires installation of PyTorch from `here <https://pytorch.org/get-started/locally/>`_.

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

