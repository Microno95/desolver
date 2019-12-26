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

import numpy
import pyaudi
import scipy
import scipy.special

# Datatypes
bool     = numpy.bool
float64  = pyaudi.gdual_double
vfloat64 = pyaudi.gdual_vdouble
uint8    = numpy.uint8
int16    = numpy.int16
int32    = numpy.int32
int64    = numpy.int64

float_fmts = {
    'float64': float64,
    'vfloat64': vfloat64    
}

def asarray(x):
    return array(x)

@numpy.vectorize
def to_float(x):
    if hasattr(x, 'constant_cf'):
        return (x.constant_cf)
    else:
        return float(x)

def to_type(x, dtype):
    return numpy.asanyarray(x).astype(dtype)

# Convenience Decorators
def type_reg(f):
    def _wrapped(*args, **kwargs):
        kwargs.setdefault("dtype", float_fmts[float_fmt()])
        return f(*args, **kwargs)
    _wrapped.original_function = f
    return _wrapped

# Fundamental Mathematical Operators
# neg      = numpy.negative
def neg(x1, *args, **kwargs):
    return -x1
# pow      = numpy.power
def pow(x1, x2, *args, **kwargs):
    return x1**x2
abs      = numpy.vectorize(pyaudi.abs)
sqrt     = numpy.vectorize(pyaudi.sqrt)

exp      = numpy.vectorize(pyaudi.exp)
# expm1    = numpy.expm1
@numpy.vectorize
def expm1(x1, *args, **kwargs):
    return exp(x1, *args, **kwargs) - 1
log      = numpy.vectorize(pyaudi.log)
# log10    = numpy.log10
@numpy.vectorize
def log10(x1, *args, **kwargs):
    return log(x1, *args, **kwargs) / log(10.0)
# log1p    = numpy.log1p
@numpy.vectorize
def log1p(x1, *args, **kwargs):
    return log(x1 + 1., *args, **kwargs)
# log2     = numpy.log2
@numpy.vectorize
def log2(x1, *args, **kwargs):
    return log(x1, *args, **kwargs) / log(2.0)

# add      = numpy.add
@numpy.vectorize
def add(x1, x2, *args, **kwargs):
    return x1 + x2
# sub      = numpy.subtract
@numpy.vectorize
def sub(x1, x2, *args, **kwargs):
    return x1 - x2
# div      = numpy.divide
@numpy.vectorize
def div(x1, x2, *args, **kwargs):
    return x1 / x2
# mul      = numpy.multiply
@numpy.vectorize
def mul(x1, x2, *args, **kwargs):
    return x1 * x2

# reciprocal = numpy.reciprocal
@numpy.vectorize
def reciprocal(x1, *args, **kwargs):
    return 1 / x1
# remainder  = numpy.remainder
@numpy.vectorize
def remainder(x1, x2, *args, **kwargs):
    raise NotImplementedError("Func remainder is not defined for pyaudi variables.")

# ceil     = numpy.ceil
def ceil(*args, **kwargs):
    raise NotImplementedError("Func ceil is not defined for pyaudi variables.")
# floor    = numpy.floor
def floor(*args, **kwargs):
    raise NotImplementedError("Func floor is not defined for pyaudi variables.")
# round    = numpy.round
def round(*args, **kwargs):
    raise NotImplementedError("Func round is not defined for pyaudi variables.")
# fmod     = numpy.fmod
def fmod(*args, **kwargs):
    raise NotImplementedError("Func fmod is not defined for pyaudi variables.")
    
# clip     = numpy.clip
def clip(*args, **kwargs):
    raise NotImplementedError("Func clip is not defined for pyaudi variables.")
# sign     = numpy.sign
@numpy.vectorize
def sign(x1, *args, **kwargs):
    return numpy.sign(to_float(x1), *args, **kwargs)
# trunc    = numpy.trunc
def trunc(*args, **kwargs):
    raise NotImplementedError("Func trunc is not defined for pyaudi variables.")

# Trigonometric Functions
cos      = numpy.vectorize(pyaudi.cos)
sin      = numpy.vectorize(pyaudi.sin)
tan      = numpy.vectorize(pyaudi.tan)

cosh     = numpy.vectorize(pyaudi.cosh)
sinh     = numpy.vectorize(pyaudi.sinh)
tanh     = numpy.vectorize(pyaudi.tanh)

acos     = numpy.vectorize(pyaudi.acos)
asin     = numpy.vectorize(pyaudi.asin)
atan     = numpy.vectorize(pyaudi.atan)
# atan2    = numpy.arctan2
@numpy.vectorize
def atan2(x1, x2, *args, **kwargs):
    return atan(x1/x2) + (to_float(x2) < 0) * pi
    
# Other Functions
# digamma  = scipy.special.digamma
def digamma(*args, **kwargs):
    raise NotImplementedError("Func digamma is not defined for pyaudi variables.")
erf      = numpy.vectorize(pyaudi.erf)
# erfc     = scipy.special.erfc
@numpy.vectorize
def erfc(x1, *args, **kwargs):
    return 1 - erf(x1, *args, **kwargs)
# erfinv   = scipy.special.erfinv
def erfinv(*args, **kwargs):
    raise NotImplementedError("Func erfinv is not defined for pyaudi variables.")
# sigmoid  = scipy.special.expit
@numpy.vectorize
def sigmoid(x1, *args, **kwargs):
    return 1/(1 + exp(-x1))

# Additional Definitions
def rsqrt(x, out=None):
    return pow(x, -0.5, out=out)

def addcdiv(x, value=1, y1=None, y2=None, out=None):
    if y1 is None or y2 is None:
        raise ValueError("y1 and y2 must both be specified")
    if out is None:
        out = value * div(y1, y2)
        out = x + out
    else:
        div(y1, y2, out=out)
        mul(value, out, out=out)
        add(x, out, out=out)
    return out

def addcmul(x, value=1, y1=None, y2=None, out=None):
    if y1 is None or y2 is None:
        raise ValueError("y1 and y2 must both be specified")
    if out is None:
        out = value * mul(y1, y2)
        out = x + out
    else:
        mul(y1, y2, out=out)
        mul(value, out, out=out)
        add(x, out, out=out)
    return out

def frac(x, out=None):
    if out is None:
        return x - floor(x)
    floor(x, out=out)
    sub(x, out=out)
    return out

def lerp(start, end, weight, out=None):
    if out is None:
        return start + weight * (end - start)
    sub(end, start, out=out)
    mul(weight, out, out=out)
    add(start, out, out=out)
    return out

def mvlgamma(x, p):
    raise NotImplementedError("Func mvlgamma is not defined for pyaudi variables.")

# Common Array Operations
einsum      = numpy.einsum
def einsum(*args, **kwargs):
    raise NotImplementedError("Func einsum is not defined for pyaudi variables.")
concatenate = numpy.concatenate
append      = numpy.append
stack       = numpy.stack
ravel       = numpy.ravel
flatten     = numpy.ravel
arange      = type_reg(numpy.arange)
logspace    = type_reg(numpy.logspace)
linspace    = type_reg(numpy.linspace)
eye         = type_reg(numpy.eye)

# Reduction Ops
argmax    = numpy.argmax
argmin    = numpy.argmin
cumprod   = numpy.cumprod
cumsum    = numpy.cumsum
logsumexp = scipy.special.logsumexp
mean      = numpy.mean
median    = numpy.median
prod      = numpy.prod
std       = numpy.std
var       = numpy.var
sum       = numpy.sum
# norm      = numpy.linalg.norm
def norm(x1, ord=None):
    if ord == 0:
        return sum(x != 0)
    elif ord == 1:
        return max(sum(abs(x), axis=1))
    elif ord == -1:
        return min(sum(abs(x), axis=1))
    elif isinstance(ord, int):
        return sum(abs(x1)**ord)**(1./ord)
    return sqrt(sum(abs(x1)**2))

def dist(x, y, ord=2):
    return norm(x-y, ord=ord)

# Comparison Ops
allclose   = numpy.allclose
argsort    = numpy.argsort

eq         = numpy.equal
ne         = numpy.not_equal
ge         = numpy.greater_equal
gt         = numpy.greater
le         = numpy.less_equal
lt         = numpy.less

def equal(*args, **kwargs):
    return numpy.all(eq(*args, **kwargs))

isfinite   = numpy.isfinite
isinf      = numpy.isinf
isnan      = numpy.isnan
max        = numpy.max
min        = numpy.min
any        = numpy.any
all        = numpy.all

array      = type_reg(numpy.array)
zeros      = type_reg(numpy.zeros)
empty      = type_reg(numpy.empty)
full       = type_reg(numpy.full)
zeros_like = type_reg(numpy.zeros_like)
ones_like  = type_reg(numpy.ones_like)
empty_like = type_reg(numpy.empty_like)
full_like  = type_reg(numpy.full_like)

def to_numpy(x):
    return numpy.asarray(x)

def as_bool_array(x):
    return numpy.asarray(x).astype(bool)

def copy(x):
    return numpy.copy(x)

def reshape(x, new_dims):
    return numpy.reshape(asarray(x), new_dims)

def shape(x):
    return numpy.shape(x)

def logical_not(x, out=None, where=True):
    return numpy.logical_not(x, out=out, where=where)

def logical_or(a, b, out=None, where=True):
    return numpy.logical_or(a, b, out=out, where=where)

def logical_and(a, b, out=None, where=True):
    return numpy.logical_and(a, b, out=out, where=where)

def logical_xor(a, b, out=None, where=True):
    return numpy.logical_xor(a, b, out=out, where=where)

nonzero = numpy.nonzero
argsort = numpy.argsort