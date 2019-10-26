"""
The MIT License (MIT)

Copyright (c) 2019 Microno95, Ekin Ozturk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy

from .integrator_template import named_integrator
from .integrator_types import ExplicitIntegrator, SymplecticIntegrator

__all__ = [
    'RK45CKSolver',
    'RK5Solver',
    'MidpointSolver',
    'HeunsSolver',
    'EulerSolver',
    'EulerTrapSolver',
    'HeunEulerSolver',
    'SymplecticEulerSolver',
    'BABs9o7HSolver',
    'ABAs5o6HSolver'
]

@named_integrator("Explicit RK45CK",
                       alt_names=("RK45CK", "Runge-Kutta-Cash-Karp", "RK45"),
                       order=4.0)
class RK45CKSolver(ExplicitIntegrator):
    # Based on RK45 Cash-Karp
    tableau = numpy.array(
        [[0.0,  0.0,        0.0,     0.0,       0.0,          0.0,      0.0],
         [1/5,  1/5,        0.0,     0.0,       0.0,          0.0,      0.0],
         [3/10, 3/40,       9/40,    0.0,       0.0,          0.0,      0.0],
         [3/5,  3/10,       -9/10,   6/5,       0.0,          0.0,      0.0],
         [1,    -11/54,     5/2,     -70/27,    35/27,        0.0,      0.0],
         [7/8,  1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0.0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0., 37/378,     0, 250/621,     125/594,     0,         512/1771],
         [0., 2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4     ]], dtype=numpy.float64
    )
    
@named_integrator("Explicit RK5",
                   alt_names=("RK5", "Runge-Kutta 5", "RK5"),
                   order=5.0)
class RK5Solver(ExplicitIntegrator):
    # The 5th order integrator from RK45 Cash-Karp
    tableau = numpy.copy(RK45CKSolver.tableau)

    final_state = numpy.array(
        [[0., 2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4     ]], dtype=numpy.float64
    )

@named_integrator("Explicit Midpoint",
                       alt_names=("Midpoint",),
                       order=2.0)
class MidpointSolver(ExplicitIntegrator):
    tableau = numpy.array(
        [[0,    0,   0],
         [1/2,  1/2, 0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    0,   1]], dtype=numpy.float64
    )

@named_integrator("Explicit Heun's",
                       alt_names=("Heun's",),
                       order=2.0)
class HeunsSolver(ExplicitIntegrator):
    tableau = numpy.array(
        [[0,    0,   0  ],
         [1,    1,   0  ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    1/4, 3/4]], dtype=numpy.float64
    )

@named_integrator("Explicit Euler",
                       alt_names=("Forward Euler", "Euler"),
                       order=1.0)
class EulerSolver(ExplicitIntegrator):
    tableau = numpy.array(
        [[0,    0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    1]], dtype=numpy.float64
    )

@named_integrator("Explicit Euler-Trapezoidal",
                       alt_names=("Euler-Trapezoidal", "Euler-Trap", "Predictor-Corrector Euler"),
                       order=3.0)
class EulerTrapSolver(ExplicitIntegrator):
    tableau = numpy.array(
        [[0,   0,   0,     0,   0  ],
         [1,   1,   0,     0,   0  ],
         [1,   1/2, 1/2,   0,   0  ],
         [1,   1/2, 0,     1/2, 0  ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,   1/2, 0,    0,   1/2]], dtype=numpy.float64
    )

@named_integrator("Explicit Adaptive Heun-Euler",
                       alt_names=("Adaptive Heun-Euler", "AHE"),
                       order=1.0)
class HeunEulerSolver(ExplicitIntegrator):
    tableau = numpy.array(
        [[0,   0,   0],
         [1,   1,   0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    1/2, 1/2],
         [0,    1,   0]], dtype=numpy.float64
    )

@named_integrator("Explicit Symplectic Forward Euler",
                       alt_names=("Symplectic Euler",),
                       order=1.0)
class SymplecticEulerSolver(SymplecticIntegrator):
    tableau = numpy.array(
        [[0.5, 0,   0.5],
         [0,   1.0, 0  ],
         [0.5, 0,   0.5]], dtype=numpy.float64
    )

@named_integrator("Explicit BABS9O7H",
                       alt_names=("BABS9O7H", "BABs9o7H"),
                       order=7.0)
class BABs9o7HSolver(SymplecticIntegrator):
    # Based on arXiv:1501.04345v2 - BAB's9o7H
    tableau = numpy.array(
       [[ 0.                  ,  0.                  ,  0.04649290043965892 ],
        [ 0.                  ,  0.1289555065927298  ,  0.                  ],
        [ 0.                  ,  0.                  ,  0.154901012702888   ],
        [ 0.                  ,  0.10907642985488271 ,  0.                  ],
        [ 0.                  ,  0.                  ,  0.31970548287359174 ],
        [ 0.                  , -0.013886035680471514,  0.                  ],
        [ 0.                  ,  0.                  , -0.19292000881571322 ],
        [ 0.                  ,  0.18375497456418036 ,  0.                  ],
        [ 0.                  ,  0.                  ,  0.17182061279957458 ],
        [ 0.                  ,  0.18419824933735726 ,  0.                  ],
        [ 0.                  ,  0.                  ,  0.17182061279957458 ],
        [ 0.                  ,  0.18375497456418036 ,  0.                  ],
        [ 0.                  ,  0.                  , -0.19292000881571322 ],
        [ 0.                  , -0.013886035680471514,  0.                  ],
        [ 0.                  ,  0.                  ,  0.31970548287359174 ],
        [ 0.                  ,  0.10907642985488271 ,  0.                  ],
        [ 0.                  ,  0.                  ,  0.154901012702888   ],
        [ 0.                  ,  0.1289555065927298  ,  0.                  ],
        [ 1.                  ,  0.                  ,  0.04649290043965892 ]], dtype=numpy.float64
    )

@named_integrator("Explicit ABAS5O6H",
                       alt_names=("ABAS5O6H", "ABAs5o6H"),
                       order=6.0)
class ABAs5o6HSolver(SymplecticIntegrator):
    # Based on arXiv:1501.04345v2 - ABAs5o6H
    tableau = numpy.array(
      [[ 0.                  ,  0.                  ,  0.15585935917621682 ],
       [ 0.                  , -0.6859195549562167  ,  0.                  ],
       [ 0.                  ,  0.                  , -0.007025499091957318],
       [ 0.                  ,  0.9966295909529364  ,  0.                  ],
       [ 0.                  ,  0.                  ,  0.35116613991574047 ],
       [ 0.                  ,  0.3785799280065607  ,  0.                  ],
       [ 0.                  ,  0.                  ,  0.35116613991574047 ],
       [ 0.                  ,  0.9966295909529364  ,  0.                  ],
       [ 0.                  ,  0.                  , -0.007025499091957318],
       [ 0.                  , -0.6859195549562167  ,  0.                  ],
       [ 1.                  ,  0.                  ,  0.15585935917621682 ]]
   )
