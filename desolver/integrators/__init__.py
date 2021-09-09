from .integrator_template import *
from .integrator_types import *
from .explicit_integration_schemes import *
from .implicit_integration_schemes import *

__available_methods = dict()

__explicit_integration_methods__ = [
    RK1412Solver,
    RK108Solver,
    RK8713MSolver,
    RK45CKSolver,
    RK5Solver,
    RK4Solver,
    MidpointSolver,
    HeunsSolver,
    EulerSolver,
    EulerTrapSolver,
    HeunEulerSolver,
    SymplecticEulerSolver,
    BABs9o7HSolver,
    ABAs5o6HSolver,
    DOPRI45
]

__implicit_integration_methods__ = [
    GaussLegendre4,
    GaussLegendre6,
    BackwardEuler,
    ImplicitMidpoint,
    LobattoIIIA2,
    LobattoIIIA4,
    LobattoIIIB2,
    LobattoIIIB4,
    LobattoIIIC2,
    LobattoIIIC4,
    CrankNicolson,
#     DIRK3LStable,
    RadauIA3,
    RadauIA5,
    RadauIIA3,
    RadauIIA5,
    RadauIIA19
]

__available_methods.update(dict(
    [(func.__name__, func) for func in __explicit_integration_methods__ if hasattr(func, "__alt_names__")] +
    [(alt_name, func)      for func in __explicit_integration_methods__ if hasattr(func, "__alt_names__") for alt_name in func.__alt_names__]))

__available_methods.update(dict(
    [(func.__name__, func) for func in __implicit_integration_methods__ if hasattr(func, "__alt_names__")] +
    [(alt_name, func)      for func in __implicit_integration_methods__ if hasattr(func, "__alt_names__") for alt_name in func.__alt_names__]))

def available_methods(names=True):
    if names:
        return sorted(set(__available_methods.keys()))
    else:
        return __available_methods
