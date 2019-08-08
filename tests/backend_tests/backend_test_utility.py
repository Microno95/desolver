from __future__ import absolute_import, division, print_function, unicode_literals

import os

def test_backend(backend):
    try:
        os.environ['DES_BACKEND'] = backend
        import desolver as de
        import desolver.backend as D
        import numpy as np
        import scipy
        
        if backend not in ['torch']:
            # Default datatype test
            for i in D.available_float_fmt():
                D.set_float_fmt(i)
                assert(D.array(1.0).dtype  == np.dtype(D.float_fmt()))

        # Test Function Evals
        for i in D.available_float_fmt():
            D.set_float_fmt(i)
            de.deutil.warning("Testing float format {}".format(D.float_fmt()))
            pi = D.to_float(D.pi)
            
            assert(np.pi - 2*D.epsilon()          <= pi                        <= np.pi + 2*D.epsilon())
            assert(np.e - 2*D.epsilon()           <= D.to_float(D.e)           <= np.e + 2*D.epsilon())
            assert(np.euler_gamma - 2*D.epsilon() <= D.to_float(D.euler_gamma) <= np.euler_gamma + 2*D.epsilon())
            
            assert(-2*D.epsilon() <= D.sin(pi) <= 2*D.epsilon())
            assert(-2*D.epsilon() <= D.cos(pi)+1 <= 2*D.epsilon())
            assert(-2*D.epsilon() <= D.tan(pi) <= 2*D.epsilon())

            assert(D.asin(D.to_float(1)) == pi/2)
            assert(D.acos(D.to_float(1)) == 0)
            assert(D.atan(D.to_float(1)) == pi/4)
            assert(D.atan2(D.to_float(1), D.to_float(1)) == pi/4)

            assert(D.sinh(pi)        == np.sinh(pi))
            assert(D.cosh(pi)        == np.cosh(pi))
            assert(D.tanh(pi)        == np.tanh(pi))

            assert(-3.141592653589793 - 2*D.epsilon()  <= D.neg(pi)   <= -3.141592653589793 + 2*D.epsilon())
            assert(31.00627668029982 - 10*D.epsilon()  <= D.pow(pi,3) <= 31.00627668029982 + 10*D.epsilon())
            assert(3.141592653589793 - 2*D.epsilon()   <= D.abs(pi)   <= 3.141592653589793 + 2*D.epsilon())
            assert(1.77245385090551603 - 2*D.epsilon() <= D.sqrt(pi)  <= 1.77245385090551603 + 2*D.epsilon())
            assert(23.1406926327792690 - 10*D.epsilon()<= D.exp(pi)   <= 23.1406926327792690 + 10*D.epsilon())
            assert(22.1406926327792690 - 10*D.epsilon()<= D.expm1(pi) <= 22.1406926327792690 + 10*D.epsilon())
            assert(1.14472988584940017 - 2*D.epsilon() <= D.log(pi)   <= 1.14472988584940017 + 2*D.epsilon())
            assert(1.14472988584940017 - 2*D.epsilon() <= D.log(pi)   <= 1.14472988584940017 + 2*D.epsilon())
            assert(0.49714987269413385 - 2*D.epsilon() <= D.log10(pi) <= 0.49714987269413385 + 2*D.epsilon())
            assert(1.42108041279429263 - 2*D.epsilon() <= D.log1p(pi) <= 1.42108041279429263 + 2*D.epsilon())
            assert(1.65149612947231880 - 2*D.epsilon() <= D.log2(pi)  <= 1.65149612947231880 + 2*D.epsilon())

            assert(4.14159265358979324 - 2*D.epsilon() <= D.add(pi,1) <= 4.14159265358979324 + 2*D.epsilon())
            assert(2.14159265358979324 - 2*D.epsilon() <= D.sub(pi,1) <= 2.14159265358979324 + 2*D.epsilon())
            assert(D.div(pi,1)       == pi)
            assert(D.mul(pi,1)       == pi)

            assert(0.31830988618379067 - 2*D.epsilon() <= D.reciprocal(pi)  <= 0.31830988618379067 + 2*D.epsilon())
            assert(0.14159265358979324 - 2*D.epsilon() <= D.remainder(pi,3) <= 0.14159265358979324 + 2*D.epsilon())

            assert(D.ceil(pi)        == 4)
            assert(D.floor(pi)       == 3)
            assert(D.round(pi)       == 3)
            assert(1.1415926535897931 - 2*D.epsilon()  <= D.fmod(pi,2) <= 1.1415926535897931 + 2*D.epsilon())

            assert(D.clip(pi,1,2)    == 2)
            assert(D.sign(pi)        == 1)
            assert(D.trunc(pi)       == 3)

            assert(0.9772133079420067 - 2*D.epsilon()    <= D.digamma(pi)             <= 0.9772133079420067 + 2*D.epsilon())
            assert(0.9999911238536324 - 2*D.epsilon()    <= D.erf(pi)                 <= 0.9999911238536324 + 2*D.epsilon())
            assert(8.8761463676416054e-6 - 2*D.epsilon() <= D.erfc(pi)                <= 8.8761463676416054e-6 + 2*D.epsilon())
            assert(0.4769362762044699 - 2*D.epsilon()    <= D.erfinv(D.to_float(0.5)) <= 0.4769362762044699 + 2*D.epsilon())
            assert(0.9585761678336372 - 2*D.epsilon()    <= D.sigmoid(pi)             <= 0.9585761678336372 + 2*D.epsilon())

            assert(0.5641895835477563 - 2*D.epsilon()    <= D.rsqrt(pi)               <= 0.5641895835477563 + 2*D.epsilon())
            assert(pi + 0.5 - 2*D.epsilon()              <= D.lerp(pi,pi+1,0.5)       <= pi + 0.5 + 2*D.epsilon())
            assert(1.7891115385869942 - 2*D.epsilon()    <= D.mvlgamma(pi, 2)         <= 1.7891115385869942 + 2*D.epsilon())
            
            assert(D.addcdiv(pi,1,D.to_float(3),D.to_float(2))   == pi + (1 * (3 / 2)))
            assert(D.addcmul(pi,1,D.to_float(3),D.to_float(2))   == pi + (1 * (3 * 2)))
            assert(D.frac(pi)            == pi - 3)
    except:
        print("{} Backend Test Failed".format(os.environ['DES_BACKEND']))
        raise
    print("{} Backend Test Succeeded".format(os.environ['DES_BACKEND']))
    

# # Common Array Operations
# einsum      = numpy.einsum
# concatenate = numpy.concatenate
# append      = numpy.append
# stack       = numpy.stack
# ravel       = numpy.ravel
# flatten     = numpy.ravel
# arange      = type_reg(numpy.arange)
# logspace    = type_reg(numpy.logspace)
# linspace    = type_reg(numpy.linspace)
# eye         = type_reg(numpy.eye)

# # Reduction Ops
# argmax    = numpy.argmax
# argmin    = numpy.argmin
# cumprod   = numpy.cumprod
# cumsum    = numpy.cumsum
# logsumexp = scipy.special.logsumexp
# mean      = numpy.mean
# median    = numpy.median
# prod      = numpy.prod
# std       = numpy.std
# var       = numpy.var
# sum       = numpy.sum
# norm      = numpy.linalg.norm

# def dist(x, y, ord=2):
#     return numpy.linalg.norm(x-y, ord=ord)

# # Comparison Ops
# allclose   = numpy.allclose
# argsort    = numpy.argsort

# eq         = numpy.equal
# ne         = numpy.not_equal
# ge         = numpy.greater_equal
# gt         = numpy.greater
# le         = numpy.less_equal
# lt         = numpy.less

# def equal(*args, **kwargs):
#     return numpy.all(eq(*args, **kwargs))

# isfinite   = numpy.isfinite
# isinf      = numpy.isinf
# isnan      = numpy.isnan
# max        = numpy.max
# min        = numpy.min

# array      = type_reg(numpy.array)
# zeros      = type_reg(numpy.zeros)
# empty      = type_reg(numpy.empty)
# full       = type_reg(numpy.full)
# zeros_like = type_reg(numpy.zeros_like)
# ones_like  = type_reg(numpy.ones_like)
# empty_like = type_reg(numpy.empty_like)
# full_like  = type_reg(numpy.full_like)