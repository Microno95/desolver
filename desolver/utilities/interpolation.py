from desolver import backend as D

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
        self.t1 = D.ar_numpy.copy(t1)
        self.t0 = D.ar_numpy.copy(t0)
        self.p0 = D.ar_numpy.copy(p0)
        self.p1 = D.ar_numpy.copy(p1)
        self.m0 = D.ar_numpy.copy(m0)
        self.m1 = D.ar_numpy.copy(m1)
        
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
        t = self.__affine_transform(t_eval)
        if t == 0.0:
            return self.p0
        elif t == 1.0:
            return self.p1
        t2 = t**2
        t3 = t2 * t

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        return h00 * self.p0 + h10 * self.trange * self.m0 + h01 * self.p1 + h11 * self.trange * self.m1

    def grad(self, t_eval):
        t = t_eval
        t_aff = (t - self.tshift)/self.trange
        if t_aff == 0.0:
            return self.m0
        elif t_aff == 1.0:
            return self.m1
        t2 = 2 * (t - self.tshift)/self.trange * (1/self.trange)
        t3 = 3 * (t - self.tshift)/self.trange * (t - self.tshift)/self.trange * (1/self.trange)

        h00 = 2 * t3 - 3 * t2
        h10 = t3 - 2 * t2 + (1/self.trange)
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        return h00 * self.p0 + h10 * self.trange * self.m0 + h01 * self.p1 + h11 * self.trange * self.m1

    def __repr__(self):
        return f"<CubicHermiteInterp(t0={self.t0}, t1={self.t1}, |p0|={D.ar_numpy.linalg.norm(self.p0)}, |dp|={D.ar_numpy.linalg.norm(self.p0 - self.p1)}, " \
               f"|p1|={D.ar_numpy.linalg.norm(self.p1)}, |m0|={D.ar_numpy.linalg.norm(self.m0)}, |m1|={D.ar_numpy.linalg.norm(self.m1)}, |dm|={D.ar_numpy.linalg.norm(self.m0 - self.m1)})>"
    

# class FactorFunction:
#     """Function Factoring Class

#     Given a function f, this class acts a wrapper that enables computing the function f
#     with the roots at r_i factored out. 
    
#     With the following assumptions:
#     - The roots are assumed to have finite multiplicity
#     - The function f is piecewise continuous.
    
#     One can factor out the root (x - r_i)**m where m is the multiplicity
#     by computing f(x)/Prod(x - r_i, i={0, n})**m when x is far from r_i and 
#     (d/dx)^(m) f(x)/Prod(x - r_j, j={0,n}, j != i) when x is near the root.
    
#     This follows from L'Hopital's Rule which states that if the limit of f(x)/g(x) as
#     x->r diverges OR either the numerator or denominator goes to infinity, then the limit
#     as x->r is equal to the limit of the ratio of the derivatives of f(x) and g(x).
    
#     In our case g(x) = (x - r_i)**m, thus applying L'Hopital's Rule m-times yields a
#     finite value.
    
#     The caveats are that the values computed by this function are not necessarily 
#     correct when x is close to r_i and the values may be ill-conditioned 
#     near r_i (not necessarily code breaking for root-finding).
    
#     For a finite polynomial f(x) with roots at r_i, this class
#     will factor out the monomials associated with the roots
#     and, critically, will have the correct factored values at
#     f(r_i)/(x - r_i)**m.
    
#     Parameters
#     ----------
#     f : callable
#         Univariate function to factor out the roots of
#     roots : D.ar_numpy.array
#         Array of roots values to factor out from the function f(x)
#     """
#     def __init__(self, f, roots):
#         self.f = f
#         self.roots = D.ar_numpy.array(roots)
    
#     def j(self, f, x, nu=1):
#         __j = utilities.JacobianWrapper(f)
#         for i in range(nu-1):
#             __j = utilities.JacobianWrapper(__j)
#         return __j(x)
    
#     def __call__(self, x):
#         x = D.ar_numpy.asarray(x)
#         if len(D.ar_numpy.shape(x)) > 1:
#             return self(x.reshape(-1)).reshape(D.ar_numpy.shape(x))
#         elif len(D.ar_numpy.shape(x)) == 1:
#             return D.ar_numpy.stack([self(i) for i in x])
#         else:
#             if len(self.roots) == 0:
#                 return self.f(x)
#             if D.autoray.infer_backend(x) == 'torch':
#                 if not x.requires_grad:
#                     x.requires_grad = True
#             d_monomials = x - self.roots
#             d_zeros     = D.ar_numpy.abs(d_monomials) <= 64*D.epsilon(x.dtype)
#             num_zeros   = len(D.ar_numpy.nonzero(d_zeros)[0])
#             if num_zeros > 0:
#                 if D.autoray.infer_backend(x) == 'torch':
#                     denom       = D.ar_numpy.ones_like(x)
#                     for mono in d_monomials:
#                         denom  *= mono
#                     numerator   = D.ar_numpy.jacobian(self.f(x), x, nu=num_zeros)
#                     denominator = D.ar_numpy.jacobian(denom, x, nu=num_zeros)
#                 else:
#                     numerator   = self.j(self.f, x, nu=num_zeros)
#                     denominator = self.j(lambda y: D.ar_numpy.prod(y - self.roots), x, nu=num_zeros)
#             else:
#                 numerator   = self.f(x)
#                 denominator = D.ar_numpy.prod(d_monomials)
#             out = numerator / denominator
#             return out