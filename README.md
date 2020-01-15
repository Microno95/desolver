# DESolver
[![BCH compliance](https://bettercodehub.com/edge/badge/Microno95/desolver?branch=master)](https://bettercodehub.com/)
[![Build Status](https://travis-ci.com/Microno95/desolver.svg?branch=master)](https://travis-ci.com/Microno95/desolver)
[![codecov](https://codecov.io/gh/Microno95/desolver/branch/master/graph/badge.svg)](https://codecov.io/gh/Microno95/desolver)

This is a python package for solving Initial Value Problems using various numerical integrators.
Many integration routines are included ranging from fixed step to symplectic to adaptive integrators.

Implicit integrators are intended for release 4.0, but that's far off for now.

# In Beta Development
**3.0.0b12** - PyAudi support has been added to the module. It is now possible to do numerical integrations using `gdual` variables such as `gdual_double`, `gdual_vdouble` and `gdual_real128` (only on select platforms, refer to [pyaudi docs](https://darioizzo.github.io/audi/) for more information).
 
This version can be installed with `pip install desolver[pyaudi]==3.0.0b12`

# Latest Release
**2.5.0** - Event detection has been added to the module. It is now possible to do numerical integration with terminal and non-terminal events.
 
**2.2.0** - PyTorch backend is now implemented. It is now possible to numerically integrate a system of equations that use pytorch tensors and then compute gradients from these.
 
**Use of PyTorch backend requires installation of PyTorch from [here](https://pytorch.org/get-started/locally/).**

# To Install:
Just type

`pip install desolver`

## Implemented Integration Methods
### Adaptive Methods
#### Explicit Methods
1. Runge-Kutta 14(12) with Feagin Coefficients [**NEW**]
2. Runge-Kutta 10(8) with Feagin Coefficients [**NEW**]
3. Runge-Kutta 8(7) with Dormand-Prince Coefficients [**NEW**]
4. Runge-Kutta 4(5) with Cash-Karp Coefficients
5. Adaptive Heun-Euler Method
#### Implicit Methods
**NOT YET IMPLEMENTED**
### Fixed Step Methods
#### Explicit Methods
1. Runge-Kutta 4 - The classic RK4 integrator
2. Runge-Kutta 5 - The 5th order integrator from RK45 with Cash-Karp Coefficients.
3. BABs9o7H Method  -- Based on arXiv:1501.04345v2 - BAB's9o7H
4. ABAs5o6HA Method -- Based on arXiv:1501.04345v2 - ABAs5o6H
5. Midpoint Method
6. Heun's Method
7. Euler's Method
8. Euler-Trapezoidal Method
#### Implicit Methods
**NOT YET IMPLEMENTED**


# Minimal Working Example

This example shows the integration of a harmonic oscillator using DESolver.

``` python
import desolver as de
import desolver.backend as D

@de.rhs_prettifier("""[vx, x]""")
def rhs(t, state, **kwargs):
    x,vx = state

    dx  = vx
    dvx = -x

    return D.array([dx, dvx])

y_init = D.array([1., 0.])

a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=1e-6, atol=1e-9)

a.show_system()

a.integrate()

print(a)

print("If the integration was successful and correct, a[0].y and a[-1].y should be near identical.")
print(a[0].y, a[-1].y)
```
