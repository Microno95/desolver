import numpy
import pyaudi
import scipy
import scipy.special

from .common import *
from .numpy_backend import *

# Datatypes
gdual_double           = pyaudi.gdual_double
gdual_vdouble          = pyaudi.gdual_vdouble

def __atan2_helper(x1, x2):
    return atan(x1/x2) + (float(x2) < 0) * pi
        
# gdual_double Definitions
try:
    gdual_double.__float__ = lambda self: self.constant_cf
    # gdual_double.__int__  = lambda self: int(self.constant_cf)
    gdual_double.__abs__   = pyaudi.abs
    gdual_double.sqrt      = pyaudi.sqrt
    gdual_double.exp       = pyaudi.exp
    gdual_double.expm1     = lambda self: pyaudi.exp(self) - 1.0
    gdual_double.log       = pyaudi.log
    gdual_double.log10     = lambda self: pyaudi.log(self) / pyaudi.log(gdual_double(10.0))
    gdual_double.log1p     = lambda self: pyaudi.log(self + 1.0)
    gdual_double.log2      = lambda self: pyaudi.log(self) / pyaudi.log(gdual_double(2.0))
    gdual_double.cos       = pyaudi.cos
    gdual_double.sin       = pyaudi.sin
    gdual_double.tan       = pyaudi.tan
    gdual_double.cosh      = pyaudi.cosh
    gdual_double.sinh      = pyaudi.sinh
    gdual_double.tanh      = pyaudi.tanh
    gdual_double.arccos    = pyaudi.acos
    gdual_double.arcsin    = pyaudi.asin
    gdual_double.arctan    = pyaudi.atan
    gdual_double.arctan2   = lambda self, x2: pyaudi.__atan2_helper(self, x2)
    gdual_double.erf       = pyaudi.erf
    gdual_double.erfc      = lambda self: 1.0 - pyaudi.erf(self)

    float_fmts.update({
        'gdual_double':  gdual_double,
    })
except Exception as e:
    print("Unable to load pyaudi.gdual_double", file=sys.stderr)
    print("\t\tError raised is:", file=sys.stderr)
    print(e, file=sys.stderr)
    pass

# gdual_vdouble Definitions
try:
    gdual_vdouble.__float__ = lambda self: self.constant_cf
    # gdual_vdouble.__int__ = lambda self: int(self.constant_cf)
    gdual_vdouble.__abs__   = pyaudi.abs
    gdual_vdouble.sqrt      = pyaudi.sqrt
    gdual_vdouble.exp       = pyaudi.exp
    gdual_vdouble.expm1     = lambda self: pyaudi.exp(self) - 1.0
    gdual_vdouble.log       = pyaudi.log
    gdual_vdouble.log10     = lambda self: pyaudi.log(self) / pyaudi.log(gdual_vdouble(10.0))
    gdual_vdouble.log1p     = lambda self: pyaudi.log(self + 1.0)
    gdual_vdouble.log2      = lambda self: pyaudi.log(self) / pyaudi.log(gdual_vdouble(2.0))
    gdual_vdouble.cos       = pyaudi.cos
    gdual_vdouble.sin       = pyaudi.sin
    gdual_vdouble.tan       = pyaudi.tan
    gdual_vdouble.cosh      = pyaudi.cosh
    gdual_vdouble.sinh      = pyaudi.sinh
    gdual_vdouble.tanh      = pyaudi.tanh
    gdual_vdouble.arccos    = pyaudi.acos
    gdual_vdouble.arcsin    = pyaudi.asin
    gdual_vdouble.arctan    = pyaudi.atan
    gdual_vdouble.arctan2   = lambda self, x2: pyaudi.__atan2_helper(self, x2)
    gdual_vdouble.erf       = pyaudi.erf
    gdual_vdouble.erfc      = lambda self: 1.0 - pyaudi.erf(self)

    float_fmts.update({
        'gdual_vdouble':  gdual_vdouble,
    })
except Exception as e:
    print("Unable to load pyaudi.gdual_vdouble", file=sys.stderr)
    print("\t\tError raised is:", file=sys.stderr)
    print(e, file=sys.stderr)
    pass


def to_float(x):
    if isinstance(numpy.atleast_1d(numpy.asanyarray(x))[0], gdual_vdouble):
        if numpy.asanyarray(x).ndim > 0:
            return numpy.stack([i.constant_cf for i in numpy.asanyarray(x)]).astype(float64)
        else:
            return numpy.array(numpy.asanyarray(x).item().constant_cf).astype(float64)
    else:
        return numpy.asanyarray(x).astype(float64)


def sign(x1, *args, **kwargs):
    if asarray(x1).dtype == object:
        return numpy.sign(asarray(x1).astype(float64), *args, **kwargs)
    else:
        return numpy.sign(asarray(x1), *args, **kwargs)

def sigmoid(x1, *args, **kwargs):
    return 1/(1 + exp(-x1))

# Reduction Ops
def logsumexp(x1, *args, **kwargs):
    if x1.dtype == object:
        return log(sum(exp(x1)))
    else:
        return scipy.special.logsumexp(x1, *args, **kwargs)


# gdual_real128 Definitions
try:
    gdual_real128             = pyaudi.gdual_real128
    gdual_real128.__float__   = lambda self: float(repr(self.constant_cf))
#    gdual_real128.__int__     = lambda self: int(float(repr(self.constant_cf)))
    gdual_real128.__abs__     = pyaudi.abs
    gdual_real128.sqrt        = pyaudi.sqrt
    gdual_real128.exp         = pyaudi.exp
    gdual_real128.expm1       = lambda self: pyaudi.exp(self) - 1.0
    gdual_real128.log         = pyaudi.log
    gdual_real128.log10       = lambda self: pyaudi.log(self) / pyaudi.log(gdual_real128(10.0))
    gdual_real128.log1p       = lambda self: pyaudi.log(self + 1.0)
    gdual_real128.log2        = lambda self: pyaudi.log(self) / pyaudi.log(gdual_real128(2.0))
    gdual_real128.cos         = pyaudi.cos
    gdual_real128.sin         = pyaudi.sin
    gdual_real128.tan         = pyaudi.tan
    gdual_real128.cosh        = pyaudi.cosh
    gdual_real128.sinh        = pyaudi.sinh
    gdual_real128.tanh        = pyaudi.tanh
    gdual_real128.arccos      = pyaudi.acos
    gdual_real128.arcsin      = pyaudi.asin
    gdual_real128.arctan      = pyaudi.atan
    gdual_real128.arctan2     = lambda self, x2: pyaudi.__atan2_helper(self, x2)
    gdual_real128.erf         = pyaudi.erf
    gdual_real128.erfc        = lambda self: 1.0 - pyaudi.erf(self)
    
    float_fmts.update({
        'gdual_real128':  gdual_real128,
    })
except AttributeError:
    pass
except Exception as e:
    print("Unable to load pyaudi.gdual_real128", file=sys.stderr)
    print("\t\tError raised is:", file=sys.stderr)
    print(e, file=sys.stderr)
    pass