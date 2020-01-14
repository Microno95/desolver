import numpy

from .integrator_template import named_integrator
from .integrator_types import ExplicitRungeKuttaIntegrator, ExplicitSymplecticIntegrator

__all__ = [
    'RK8713MSolver',
    'RK45CKSolver',
    'RK5Solver',
    'RK4Solver',
    'MidpointSolver',
    'HeunsSolver',
    'EulerSolver',
    'EulerTrapSolver',
    'HeunEulerSolver',
    'SymplecticEulerSolver',
    'BABs9o7HSolver',
    'ABAs5o6HSolver'
]

@named_integrator("Explicit RK8713M",
                       alt_names=("RK87", "Runge-Kutta 8(7)", "RK8713M"))
class RK8713MSolver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements the Adaptive Runge-Kutta 8(7) method using
    the coefficients defined by Dormand and Prince.
    
    References
    ----------
    [1] Prince, P.J., and J.R. Dormand. ‘High Order Embedded Runge-Kutta Formulae’. Journal of Computational and Applied Mathematics 7, no. 1 (March 1981): 67–75. https://doi.org/10.1016/0771-050X(81)90010-3.
    """
    tableau = numpy.array(
        [[0.0,                    0.0,                  0.0,      0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [1/18,                   1/18,                 0.0,      0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [1/12,                   1/48,                 1/16,     0.0,       0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [1/8,                    1/32,                 0.0,      3/32,      0.0,                      0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [5/16,                   5/16,                 0.0,     -75/64,     75/64,                    0.0,                    0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [3/8,                    3/80,                 0.0,      0.0,       3/16,                     3/20,                   0.0,                     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [59/400,                 29443841/614563906,   0.0,      0.0,       77736538/692538347,      -28693883/1125000000,    23124283/1800000000,     0.0,                     0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [93/200,                 16016141/946692911,   0.0,      0.0,       61564180/158732637,       22789713/633445777,     545815736/2771057229,   -180193667/1043307555,    0.0,                     0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [5490023248/9719169821,  39632708/573591083,   0.0,      0.0,      -433636366/683701615,     -421739975/2616292301,   100302831/723423059,     790204164/839813087,     800635310/3783071287,    0.0,                     0.0,                   0.0,                  0.0, 0.0],
         [13/20,                  246121993/1340847787, 0.0,      0.0,      -37695042795/15268766246, -309121744/1061227803,  -12992083/490766935,      6005943493/2108947869,   393006217/1396673457,    123872331/1001029789,    0.0,                   0.0,                  0.0, 0.0],
         [1201146811/1299019798, -1028468189/846180014, 0.0,      0.0,       8478235783/508512852,     1311729495/1432422823, -10304129995/1701304382, -48777925059/3047939560,  15336726248/1032824649, -45442868181/3398467696,  3065993473/597172653,  0.0,                  0.0, 0.0],
         [1,                      185892177/718116043,  0.0,      0.0,      -3185094517/667107341,    -477755414/1098053517,  -703635378/230739211,     5731566787/1027545527,   5232866602/850066563,   -4093664535/808688257,    3962137247/1805957418, 65686358/487910083,   0.0, 0.0],
         [1,                      403863854/491063109,  0.0,      0.0,      -5068492393/434740067,    -411421997/543043805,    652783627/914296604,     11173962825/925320556,  -13158990841/6184727034,  3936647629/1978049680,  -160528059/685178525,   248638103/1413531060, 0.0, 0.0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0., 13451932/455176623, 0.0, 0.0, 0.0, 0.0, -808719846/976000145, 1757004468/5645159321, 656045339/265891186, -3867574721/1518517206, 465885868/322736535,  53011238/667516719,   2/45,                 0.0],
         [0., 14005451/335480064, 0.0, 0.0, 0.0, 0.0, -59238493/1068277825, 181606767/758867731,   561292985/797845732, -1041891430/1371343529, 760417239/1151165299, 118820643/751138087, -528747749/2220607170, 1/4]], dtype=numpy.float64
    )
    
    order = 8.0

@named_integrator("Explicit RK45CK",
                       alt_names=("RK45CK", "Runge-Kutta-Cash-Karp", "RK45"))
class RK45CKSolver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements the Adaptive Runge-Kutta 4(5) method using
    the coefficients defined by Cash and Karp.
    
    References
    ----------
    [1] Cash, J. R., and Alan H. Karp. ‘A Variable Order Runge-Kutta Method for Initial Value Problems with Rapidly Varying Right-Hand Sides’. ACM Transactions on Mathematical Software 16, no. 3 (1 September 1990): 201–22. https://doi.org/10.1145/79505.79507.
    """
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
    
    order = 5.0
    
@named_integrator("Explicit RK5",
                   alt_names=("RK5", "Runge-Kutta 5", "RK5"))
class RK5Solver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements a 5th order Runge-Kutta method.
    This is simply the fifth order method embedded in RK45CK.
    
    References
    ----------
    [1] Cash, J. R., and Alan H. Karp. ‘A Variable Order Runge-Kutta Method for Initial Value Problems with Rapidly Varying Right-Hand Sides’. ACM Transactions on Mathematical Software 16, no. 3 (1 September 1990): 201–22. https://doi.org/10.1145/79505.79507.
    """
    tableau = numpy.copy(RK45CKSolver.tableau)

    final_state = numpy.array(
        [[0., 37/378,     0, 250/621,     125/594,     0,         512/1771]], dtype=numpy.float64
    )
    
    order = 5.0
    
@named_integrator("Explicit RK4",
                   alt_names=("RK4", "Runge-Kutta 4", "RK4"))
class RK4Solver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements the classic 4th order Runge-Kutta method.
    """
    tableau = numpy.array(
        [[0,   0,   0,   0, 0],
         [1/2, 1/2, 0,   0, 0],
         [1/2, 0,   1/2, 0, 0],
         [1,   0,   0,   1, 0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0., 1/6, 1/3, 1/3, 1/6]], dtype=numpy.float64
    )
    
    order = 4.0

@named_integrator("Explicit Midpoint",
                       alt_names=("Midpoint",))
class MidpointSolver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements the midpoint method.
    """
    tableau = numpy.array(
        [[0,    0,   0],
         [1/2,  1/2, 0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    0,   1]], dtype=numpy.float64
    )
    
    order = 2.0

@named_integrator("Explicit Heun's",
                       alt_names=("Heun's",))
class HeunsSolver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements Heun's method.
    """
    tableau = numpy.array(
        [[0,    0,   0  ],
         [1,    1,   0  ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    1/4, 3/4]], dtype=numpy.float64
    )
    
    order = 2.0

@named_integrator("Explicit Euler",
                       alt_names=("Forward Euler", "Euler"))
class EulerSolver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements the Euler method.
    """
    tableau = numpy.array(
        [[0,    0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    1]], dtype=numpy.float64
    )
    
    order = 1.0

@named_integrator("Explicit Euler-Trapezoidal",
                       alt_names=("Euler-Trapezoidal", "Euler-Trap", "Predictor-Corrector Euler"))
class EulerTrapSolver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements the Euler-Trapezoidal method.
    """
    tableau = numpy.array(
        [[0,   0,   0,     0,   0  ],
         [1,   1,   0,     0,   0  ],
         [1,   1/2, 1/2,   0,   0  ],
         [1,   1/2, 0,     1/2, 0  ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,   1/2, 0,    0,   1/2]], dtype=numpy.float64
    )
    
    order = 2.0

@named_integrator("Explicit Adaptive Heun-Euler",
                       alt_names=("Adaptive Heun-Euler", "AHE"))
class HeunEulerSolver(ExplicitRungeKuttaIntegrator):
    """
    The derived class that implements the adaptive Heun-Euler method.
    This is a 1st order method (Euler) with an embedded 
    2nd order method (Heun) that does adaptive timestepping.
    """
    tableau = numpy.array(
        [[0,   0,   0],
         [1,   1,   0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    1/2, 1/2],
         [0,    1,   0   ]], dtype=numpy.float64
    )
    
    order = 2.0

@named_integrator("Explicit Symplectic Forward Euler",
                       alt_names=("Symplectic Euler",),
                       order=1.0)
class SymplecticEulerSolver(ExplicitSymplecticIntegrator):
    """
    The derived class that implements the symplectic Euler method.
    
    This is the simplest symplectic integration scheme.
    """
    tableau = numpy.array(
        [[0.5, 0,   0.5],
         [0,   1.0, 0  ],
         [0.5, 0,   0.5]], dtype=numpy.float64
    )

@named_integrator("Explicit BABS9O7H",
                       alt_names=("BABS9O7H", "BABs9o7H"),
                       order=7.0)
class BABs9o7HSolver(ExplicitSymplecticIntegrator):
    """
    The derived class that implements the 7th order 
    BAB's9o7H symplectic integrator. This integrator
    is only applicable to systems that have a Hamiltonian
    that can be split such that: `H(p,q) = T(p) + V(q)`.
    
    References
    ----------
    [1] Nielsen, Kristian Mads Egeris. ‘Efficient Fourth Order Symplectic Integrators for Near-Harmonic Separable Hamiltonian Systems’. ArXiv:1501.04345 [Physics, Physics:Quant-Ph], 9 February 2015. http://arxiv.org/abs/1501.04345.
    """
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
class ABAs5o6HSolver(ExplicitSymplecticIntegrator):
    """
    The derived class that implements the 6th order 
    ABAs5o6H symplectic integrator. This integrator
    is only applicable to systems that have a Hamiltonian
    that can be split such that `H(p,q) = T(p) + V(q)`.
    
    References
    ----------
    [1] Nielsen, Kristian Mads Egeris. ‘Efficient Fourth Order Symplectic Integrators for Near-Harmonic Separable Hamiltonian Systems’. ArXiv:1501.04345 [Physics, Physics:Quant-Ph], 9 February 2015. http://arxiv.org/abs/1501.04345.
    """
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
