from .. import backend as D

__all__ = [
    'CubicHermiteInterp'
]

class CubicHermiteInterp(object):
    """Cubic Hermite Polynomial Interpolation Class

    Constructs a cubic Hermite polynomial interpolant for a function with values p0 and p1,
    and gradients m0 and m1 at t0 and t1 respectively.
    
    Parameters
    ----------
    t0, t1 : float
        Evaluation points
    p0, p1 : float or array-type
        Function values at t0 and t1
    m0, m1 : float or array-type
        Function gradients wrt. t and t0 and t1
    """
    def __init__(self, t0, t1, p0, p1, m0, m1):
        self.t1 = D.copy(t1)
        self.t0 = D.copy(t0)
        self.p0 = D.copy(p0)
        self.p1 = D.copy(p1)
        self.m0 = D.copy(m0)
        self.m1 = D.copy(m1)
        
    @property
    def trange(self):
        return self.t1 - self.t0
    
    @property
    def tshift(self):
        return self.t0
    
    def __affine_transform(self, t):
        return (t - self.tshift)/self.trange
    
    def __call__(self, t_eval):
        """
        Parameters
        ----------
        t_eval : float or array-type
           Point to evaluate interpolant at
        """
        t        = self.__affine_transform(t_eval)
        t2       = t**2
        t3       = t2 * t
        t3mt2    = t3 - t2
        p2t3m3t2 = 2 * t3mt2 - t2
        return (1 + p2t3m3t2) * self.p0 + (t3mt2 - t2 + t) * self.trange * self.m0 - p2t3m3t2 * self.p1 + t3mt2 * self.trange * self.m1