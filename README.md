# DESolver
This is a python package for solving Initial Value Problems using various numerical integrators.
Many integration routines are included ranging from fixed step to symplectic to adaptive integrators.

# To Install:
Just type
	pip install DESolver

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
5. BABs9o7H Method -- Based on arXiv:1501.04345v2 - BAB's9o7H
5. ABAs5o6HA Method -- Based on arXiv:1501.04345v2 - ABAs5o6H
#### Implicit Methods
**NOT YET IMPLEMENTED**


# Minimal Working Example

This example shows the integration of a harmonic oscillator using DESolver.

``` python
import desolver as de

@de.rhs_prettifier("""[vx, x]""")
def rhs(t, state, **kwargs):
    x,vx = state

    dx  = vx
    dvx = -x

    return de.numpy.array([dx, dvx])

y_init = de.numpy.array([1., 0.])

a = de.OdeSystem(rhs, y0=y_init, n=y_init.shape, eta=True, dense_output=True, t=(0, 2*de.numpy.pi), dt=0.01, rtol=1e-6, atol=1e-9)

a.show_system()

a.integrate()

print(a)

print("If the integration was successful and correct, a.y[0] and a.y[-1] should be near identical.")
print(a.y[0], a.y[-1])
```
