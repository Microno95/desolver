# DESolver
[![BCH compliance](https://bettercodehub.com/edge/badge/Microno95/desolver?branch=master)](https://bettercodehub.com/)

This is a python package for solving Initial Value Problems using various numerical integrators.
Many integration routines are included ranging from fixed step to symplectic to adaptive integrators.

Implicit integrators are intended for release 3.0, but that's far off for now.

# Latest Release
**2.5.0** - Event detection has been added to the library. It is now possible to do numerical integration with terminal and non-terminal events.
 
**2.2.0** - PyTorch backend is now implemented. It is now possible to numerically integrate a system of equations that use pytorch tensors and then compute gradients from these.

**Use of PyTorch backend requires installation of PyTorch from [here](https://pytorch.org/get-started/locally/).**

# To Install:
Just type

`pip install desolver`

## Implemented Integration Methods
### Adaptive Methods
#### Explicit Methods
1. Runge-Kutta 45 with Cash-Karp Coefficients
2. Adaptive Heun-Euler Method
#### Implicit Methods
**NOT YET IMPLEMENTED**
### Fixed Step Methods
#### Explicit Methods
1. Midpoint Method
2. Heun's Method
3. Euler's Method
4. Euler-Trapezoidal Method
5. BABs9o7H Method  -- Based on arXiv:1501.04345v2 - BAB's9o7H
6. ABAs5o6HA Method -- Based on arXiv:1501.04345v2 - ABAs5o6H
7. Runge-Kutta 5 - The 5th order integrator from RK45. Very accurate with fixed step size.
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
