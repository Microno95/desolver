from desolver.integrators.components import *
from desolver.integrators import utilities
from desolver import exception_types

from .integrator_template import *
from .integrator_types import *
from .explicit_integration_schemes import *
from .implicit_integration_schemes import *

__available_methods = dict()


def register_integrator(new_integrator:IntegratorTemplate):
    __available_methods.update(dict([(new_integrator.__name__, new_integrator)]))
    if hasattr(new_integrator, "__alt_names__"):
        __available_methods.update(dict([(alt_name, new_integrator) for alt_name in new_integrator.__alt_names__]))


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
    DIRK3LStable,
    RadauIA3,
    RadauIA5,
    RadauIIA3,
    RadauIIA5,
    RadauIIA19
]


for func in __explicit_integration_methods__ + __implicit_integration_methods__:
    register_integrator(func)


def available_methods(names=True)->list[IntegratorTemplate]|dict[str, IntegratorTemplate]:
    if names:
        return sorted(set(__available_methods.keys()))
    else:
        return __available_methods


def explicit_methods():
    return __explicit_integration_methods__


def implicit_methods():
    return __implicit_integration_methods__
