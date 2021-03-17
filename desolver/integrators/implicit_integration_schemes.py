import numpy

from .integrator_types import ImplicitRungeKuttaIntegrator
from .. import backend as D

class GaussLegendre4(ImplicitRungeKuttaIntegrator):
    order = 4
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.5 - numpy.sqrt(3)/6,    0.25,      0.25 - numpy.sqrt(3)/6],
         [0.5 + numpy.sqrt(3)/6,    0.25 + numpy.sqrt(3)/6,      0.25]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    0.5,    0.5]], dtype=numpy.float64
    )
    
class GaussLegendre6(ImplicitRungeKuttaIntegrator):
    order = 6
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    s = numpy.sqrt(15)
    
    tableau = numpy.array(
        [[0.5 - s/10, 5/36,        2/9 - s/15, 5/36 - s/30],
         [0.5,        5/36 + s/24, 2/9,        5/36 - s/24],
         [0.5 + s/10, 5/36 + s/30, 2/9 + s/15, 5/36       ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,  5/18, 4/9,  5/18]], dtype=numpy.float64
    )
    
    del s
    
class LobattoIIIA2(ImplicitRungeKuttaIntegrator):
    order = 2
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0,   0,   0  ],
         [1.0, 0.5, 0.5]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    0.5,    0.5]], dtype=numpy.float64
    )
    
class LobattoIIIA4(ImplicitRungeKuttaIntegrator):
    order = 4
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.0, 0.0,  0.0,  0.0 ],
         [0.5, 5/24, 1/3, -1/24],
         [1.0, 1/6,  2/3,  1/6 ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0.0, 1/6,  2/3,  1/6]], dtype=numpy.float64
    )
    
class LobattoIIIB2(ImplicitRungeKuttaIntegrator):
    order = 2
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.0,   0.5,  -0.5],
         [1.0,   0.5,   0.5]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    0.5,    0.5]], dtype=numpy.float64
    )
    
class LobattoIIIB4(ImplicitRungeKuttaIntegrator):
    order = 4
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.0, 1/6,  -1/6,  0.0 ],
         [0.5, 1/6,   1/3,  0.0 ],
         [1.0, 1/6,   5/6,  0.0 ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0.0, 1/6,  2/3,  1/6]], dtype=numpy.float64
    )
    
class LobattoIIIC2(ImplicitRungeKuttaIntegrator):
    order = 2
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.5,   0.5,   0  ],
         [0.5,   0.5,   0  ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,    0.5,    0.5]], dtype=numpy.float64
    )
    
class LobattoIIIC4(ImplicitRungeKuttaIntegrator):
    order = 4
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.0, 1/6,  -1/3,   1/6 ],
         [0.5, 1/6,   5/12, -1/12],
         [1.0, 1/6,   2/3,   1/6 ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0.0, 1/6,  2/3,  1/6]], dtype=numpy.float64
    )
    
class BackwardEuler(ImplicitRungeKuttaIntegrator):
    order = 1
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[1.0, 0.0]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,   1.0]], dtype=numpy.float64
    )
    
class ImplicitMidpoint(ImplicitRungeKuttaIntegrator):
    order = 2
    __adaptive__   = False
    __symplectic__ = True
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.5, 0.5]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,   1.0]], dtype=numpy.float64
    )
    
class CrankNicolson(ImplicitRungeKuttaIntegrator):
    order = 2
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.5, 0.5]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,   0.5, 0.5]], dtype=numpy.float64
    )
    
class DIRK3LStable(ImplicitRungeKuttaIntegrator):
    order = 3
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.5,      0.5,     0.0, 0.0, 0.0],
         [2.0/3.0,  1.0/6.0, 0.5, 0.0, 0.0],
         [0.5,     -0.5,     0.5, 0.5, 0.0],
         [1.0,      1.5,    -1.5, 0.5, 0.5]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0,   1.5, -1.5, 0.5, 0.5]], dtype=numpy.float64
    )
    
class RadauIA3(ImplicitRungeKuttaIntegrator):
    order = 3
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[0.0, 1/4, -1/4],
         [2/3, 1/4,  5/12]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0, 1/4, 3/4]], dtype=numpy.float64
    )
    
    
class RadauIA5(ImplicitRungeKuttaIntegrator):
    order = 5
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    s = numpy.sqrt(6)
    
    tableau = numpy.array(
        [[0.0,        1/9, (-1-s)/18,     (-1+s)/18      ],
         [3/5 - s/10, 1/9, 11/45+7*s/360,  11/45-43*s/360],
         [3/5 + s/10, 1/9, 11/45+43*s/360, 11/45-7*s/360 ]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0, 1/9, 4/9+s/36, 4/9-s/36]], dtype=numpy.float64
    )
    
    del s

    
class RadauIIA3(ImplicitRungeKuttaIntegrator):
    order = 3
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    tableau = numpy.array(
        [[1/3, 5/12, -1/12],
         [1, 3/4, 1/3]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0, 3/4, 1/4]], dtype=numpy.float64
    )
    
class RadauIIA5(ImplicitRungeKuttaIntegrator):
    order = 5
    __adaptive__   = False
    __symplectic__ = False
    __alt_names__  = tuple()
    
    s = numpy.sqrt(6)
    
    tableau = numpy.array(
        [[(4-s)/10, (88-7*s)/360, (296-169*s)/1800, (-2+3*s)/225],
         [(4+s)/10, (296+169*s)/1800, (88+7*s)/360, (-2-3*s)/225],
         [1, (16-s)/36, (16+s)/36, 1/9]], dtype=numpy.float64
    )

    final_state = numpy.array(
        [[0, (16-s)/36, (16+s)/36, 1/9]], dtype=numpy.float64
    )
    
    del s