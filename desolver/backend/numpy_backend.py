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
import scipy
import scipy.special

# Datatypes
bool    = numpy.bool
float16 = numpy.float16
float32 = numpy.float32
float64 = numpy.float64
uint8   = numpy.uint8
int16   = numpy.int16
int32   = numpy.int32
int64   = numpy.int64

def asarray(x):
    return array(x)

def to_float(x):
    return float(x)

def to_type(x, dtype):
    return numpy.asanyarray(x).astype(dtype)

# Convenience Decorators
def type_reg(f):
    def _wrapped(*args, **kwargs):
        kwargs.setdefault("dtype", float_fmt())
        return f(*args, **kwargs)
    _wrapped.original_function = f
    return _wrapped

# Fundamental Mathematical Operators
neg      = numpy.negative
pow      = numpy.power
abs      = numpy.abs
sqrt     = numpy.sqrt

exp      = numpy.exp
expm1    = numpy.expm1
log      = numpy.log
log10    = numpy.log10
log1p    = numpy.log1p
log2     = numpy.log2

add      = numpy.add
sub      = numpy.subtract
div      = numpy.divide
mul      = numpy.multiply

reciprocal = numpy.reciprocal
remainder  = numpy.remainder

ceil     = numpy.ceil
floor    = numpy.floor
round    = numpy.round
fmod     = numpy.fmod
    
clip     = numpy.clip
sign     = numpy.sign
trunc    = numpy.trunc

# Trigonometric Functions
cos      = numpy.cos
sin      = numpy.sin
tan      = numpy.tan

cosh     = numpy.cosh
sinh     = numpy.sinh
tanh     = numpy.tanh 

acos     = numpy.arccos
asin     = numpy.arcsin
atan     = numpy.arctan
atan2    = numpy.arctan2

# Other Functions
digamma  = scipy.special.digamma
erf      = scipy.special.erf
erfc     = scipy.special.erfc
erfinv   = scipy.special.erfinv
sigmoid  = scipy.special.expit

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
    return scipy.special.multigammaln(x, d=p)

# Common Array Operations
einsum      = numpy.einsum
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
norm      = numpy.linalg.norm

def dist(x, y, ord=2):
    return numpy.linalg.norm(x-y, ord=ord)

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