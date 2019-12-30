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

from .common import *
from .numpy_backend import *

import numpy
import pyaudi
import scipy
import scipy.special

# Datatypes
import pyaudi

gdual_double  = pyaudi.gdual_double
gdual_vdouble = pyaudi.gdual_vdouble
gdual_real128 = pyaudi.gdual_real128

float_fmts.update({
    'gdual_double':  gdual_double,
    'gdual_vdouble': gdual_vdouble,
    'gdual_real128': gdual_real128
})

gdual_double.__float__  = lambda self: self.constant_cf
gdual_vdouble.__float__ = lambda self: self.constant_cf
gdual_real128.__float__ = lambda self: float(repr(self.constant_cf))

# Fundamental Mathematical Operators
# abs      = numpy.vectorize(pyaudi.abs)
gdual_double.__abs__  = lambda self: pyaudi.abs(self)
gdual_vdouble.__abs__ = lambda self: pyaudi.abs(self)
gdual_real128.__abs__ = lambda self: pyaudi.abs(self)


# sqrt     = numpy.vectorize(pyaudi.sqrt)
gdual_double.sqrt  = lambda self: pyaudi.sqrt(self)
gdual_vdouble.sqrt = lambda self: pyaudi.sqrt(self)
gdual_real128.sqrt = lambda self: pyaudi.sqrt(self)

# exp      = numpy.vectorize(pyaudi.exp)
gdual_double.exp  = lambda self: pyaudi.exp(self)
gdual_vdouble.exp = lambda self: pyaudi.exp(self)
gdual_real128.exp = lambda self: pyaudi.exp(self)

# expm1    = numpy.expm1
gdual_double.expm1  = lambda self: pyaudi.exp(self) - 1.0
gdual_vdouble.expm1 = lambda self: pyaudi.exp(self) - 1.0
gdual_real128.expm1 = lambda self: pyaudi.exp(self) - 1.0

# log      = numpy.vectorize(pyaudi.log)
gdual_double.log  = lambda self: pyaudi.log(self)
gdual_vdouble.log = lambda self: pyaudi.log(self)
gdual_real128.log = lambda self: pyaudi.log(self)

# log10    = numpy.log10
gdual_double.log10  = lambda self: pyaudi.log(self) / pyaudi.log(gdual_double(10.0))
gdual_vdouble.log10 = lambda self: pyaudi.log(self) / pyaudi.log(gdual_vdouble(10.0))
gdual_real128.log10 = lambda self: pyaudi.log(self) / pyaudi.log(gdual_real128(10.0))

# log1p    = numpy.log1p
gdual_double.log1p  = lambda self: pyaudi.log(self + 10)
gdual_vdouble.log1p = lambda self: pyaudi.log(self + 10)
gdual_real128.log1p = lambda self: pyaudi.log(self + 10)

# log2     = numpy.log2
gdual_double.log2  = lambda self: pyaudi.log(self) / pyaudi.log(gdual_double(2.0))
gdual_vdouble.log2 = lambda self: pyaudi.log(self) / pyaudi.log(gdual_vdouble(2.0))
gdual_real128.log2 = lambda self: pyaudi.log(self) / pyaudi.log(gdual_real128(2.0))

def sign(x1, *args, **kwargs):
    if numpy.asanyarray(x1).dtype == object:
        return numpy.sign(x1.astype(float64), *args, **kwargs)
    else:
        return numpy.sign(x1, *args, **kwargs)

# Trigonometric Functions
# cos      = numpy.vectorize(pyaudi.cos)
gdual_double.cos  = lambda self: pyaudi.cos(self)
gdual_vdouble.cos = lambda self: pyaudi.cos(self)
gdual_real128.cos = lambda self: pyaudi.cos(self)

# sin      = numpy.vectorize(pyaudi.sin)
gdual_double.sin  = lambda self: pyaudi.sin(self)
gdual_vdouble.sin = lambda self: pyaudi.sin(self)
gdual_real128.sin = lambda self: pyaudi.sin(self)

# tan      = numpy.vectorize(pyaudi.tan)
gdual_double.tan  = lambda self: pyaudi.tan(self)
gdual_vdouble.tan = lambda self: pyaudi.tan(self)
gdual_real128.tan = lambda self: pyaudi.tan(self)


# cosh     = numpy.vectorize(pyaudi.cosh)
gdual_double.cosh  = lambda self: pyaudi.cosh(self)
gdual_vdouble.cosh = lambda self: pyaudi.cosh(self)
gdual_real128.cosh = lambda self: pyaudi.cosh(self)

# sinh     = numpy.vectorize(pyaudi.sinh)
gdual_double.sinh  = lambda self: pyaudi.sinh(self)
gdual_vdouble.sinh = lambda self: pyaudi.sinh(self)
gdual_real128.sinh = lambda self: pyaudi.sinh(self)

# tanh     = numpy.vectorize(pyaudi.tanh)
gdual_double.tanh  = lambda self: pyaudi.tanh(self)
gdual_vdouble.tanh = lambda self: pyaudi.tanh(self)
gdual_real128.tanh = lambda self: pyaudi.tanh(self)


# acos     = numpy.vectorize(pyaudi.acos)
gdual_double.arccos  = lambda self: pyaudi.acos(self)
gdual_vdouble.arccos = lambda self: pyaudi.acos(self)
gdual_real128.arccos = lambda self: pyaudi.acos(self)

# asin     = numpy.vectorize(pyaudi.asin)
gdual_double.arcsin  = lambda self: pyaudi.asin(self)
gdual_vdouble.arcsin = lambda self: pyaudi.asin(self)
gdual_real128.arcsin = lambda self: pyaudi.asin(self)

# atan     = numpy.vectorize(pyaudi.atan)
gdual_double.arctan  = lambda self: pyaudi.atan(self)
gdual_vdouble.arctan = lambda self: pyaudi.atan(self)
gdual_real128.arctan = lambda self: pyaudi.atan(self)

# atan2    = numpy.arctan2
def __atan2_helper(x1, x2):
    return atan(x1/x2) + (float(x2) < 0) * pi

gdual_double.arctan2  = lambda self, x2: pyaudi.__atan2_helper(self, x2)
gdual_vdouble.arctan2 = lambda self, x2: pyaudi.__atan2_helper(self, x2)
gdual_real128.arctan2 = lambda self, x2: pyaudi.__atan2_helper(self, x2)
    
# Other Functions
# erf      = numpy.vectorize(pyaudi.erf)
gdual_double.erf  = lambda self: pyaudi.erf(self)
gdual_vdouble.erf = lambda self: pyaudi.erf(self)
gdual_real128.erf = lambda self: pyaudi.erf(self)

# erfc     = scipy.special.erfc
gdual_double.erfc  = lambda self: 1.0 - pyaudi.erf(self)
gdual_vdouble.erfc = lambda self: 1.0 - pyaudi.erf(self)
gdual_real128.erfc = lambda self: 1.0 - pyaudi.erf(self)

# erfinv   = scipy.special.erfinv
# sigmoid  = scipy.special.expit
def sigmoid(x1, *args, **kwargs):
    return 1/(1 + exp(-x1))

# Additional Definitions
# def rsqrt(x, out=None):
#     return pow(x, -0.5, out=out)

# def addcdiv(x, value=1, y1=None, y2=None, out=None):
#     if y1 is None or y2 is None:
#         raise ValueError("y1 and y2 must both be specified")
#     if out is None:
#         out = value * div(y1, y2)
#         out = x + out
#     else:
#         div(y1, y2, out=out)
#         mul(value, out, out=out)
#         add(x, out, out=out)
#     return out

# def addcmul(x, value=1, y1=None, y2=None, out=None):
#     if y1 is None or y2 is None:
#         raise ValueError("y1 and y2 must both be specified")
#     if out is None:
#         out = value * mul(y1, y2)
#         out = x + out
#     else:
#         mul(y1, y2, out=out)
#         mul(value, out, out=out)
#         add(x, out, out=out)
#     return out

# def frac(x, out=None):
#     if out is None:
#         return x - floor(x)
#     floor(x, out=out)
#     sub(x, out=out)
#     return out

# def lerp(start, end, weight, out=None):
#     if out is None:
#         return start + weight * (end - start)
#     sub(end, start, out=out)
#     mul(weight, out, out=out)
#     add(start, out, out=out)
#     return out

# Common Array Operations
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

# Reduction Ops
# argmax    = numpy.argmax
# argmin    = numpy.argmin
# cumprod   = numpy.cumprod
# cumsum    = numpy.cumsum
def logsumexp(x1, *args, **kwargs):
    if x1.dtype == object:
        return log(sum(exp(x1)))
    else:
        return scipy.special.logsumexp(x1, *args, **kwargs)

# mean      = numpy.mean
# median    = numpy.median
# prod      = numpy.prod
# std       = numpy.std
# var       = numpy.var
# sum       = numpy.sum
# norm      = numpy.linalg.norm
# def norm(x1, ord=None):
#     if ord == 0:
#         return sum(x != 0)
#     elif ord == 1:
#         return max(sum(abs(x), axis=1))
#     elif ord == -1:
#         return min(sum(abs(x), axis=1))
#     elif isinstance(ord, int):
#         return sum(abs(x1)**ord)**(1./ord)
#     return sqrt(sum(abs(x1)**2))

# def dist(x, y, ord=2):
#     return norm(x-y, ord=ord)

# Comparison Ops
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
# any        = numpy.any
# all        = numpy.all

# array      = type_reg(numpy.array)
# zeros      = type_reg(numpy.zeros)
# empty      = type_reg(numpy.empty)
# full       = type_reg(numpy.full)
# zeros_like = type_reg(numpy.zeros_like)
# ones_like  = type_reg(numpy.ones_like)
# empty_like = type_reg(numpy.empty_like)
# full_like  = type_reg(numpy.full_like)

# def to_numpy(x):
#     return numpy.asarray(x)

# def as_bool_array(x):
#     return numpy.asarray(x).astype(bool)

# def copy(x):
#     return numpy.copy(x)

# def reshape(x, new_dims):
#     return numpy.reshape(asarray(x), new_dims)

# def shape(x):
#     return numpy.shape(x)

# def logical_not(x, out=None, where=True):
#     return numpy.logical_not(x, out=out, where=where)

# def logical_or(a, b, out=None, where=True):
#     return numpy.logical_or(a, b, out=out, where=where)

# def logical_and(a, b, out=None, where=True):
#     return numpy.logical_and(a, b, out=out, where=where)

# def logical_xor(a, b, out=None, where=True):
#     return numpy.logical_xor(a, b, out=out, where=where)

# nonzero = numpy.nonzero
# argsort = numpy.argsort