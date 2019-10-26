from __future__ import absolute_import, division, print_function, unicode_literals

from .integrator_template import *
from .integrator_types import *
from .integration_schemes import *

available_methods = dict()

__integration_methods__ = [
    RK45CKSolver,
    RK5Solver,
    MidpointSolver,
    HeunsSolver,
    EulerSolver,
    EulerTrapSolver,
    HeunEulerSolver,
    SymplecticEulerSolver,
    BABs9o7HSolver,
    ABAs5o6HSolver
]

available_methods.update(dict([(func.__name__, func) for func in __integration_methods__ if hasattr(func, "__alt_names__")] +
                              [(alt_name, func) for func in __integration_methods__ if hasattr(func, "__alt_names__") for alt_name in func.__alt_names__]))

