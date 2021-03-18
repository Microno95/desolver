import numpy
import pyaudi
import scipy
import scipy.special

from .common import *
from .numpy_backend import *

# Datatypes
gdual_double = pyaudi.gdual_double
gdual_vdouble = pyaudi.gdual_vdouble


def __atan2_helper(x1, x2):
    if isinstance(x2, gdual_vdouble):
        return atan(x1 / x2) + (gdual_vdouble(list(map(lambda x: (x < 0) * pi, x2.constant_cf))))
    else:
        return atan(x1 / x2) + (float(x2) < 0) * pi


# gdual_double Definitions
gdual_double.__float__ = lambda self: self.constant_cf
# gdual_double.__int__  = lambda self: int(self.constant_cf)
gdual_double.__abs__ = lambda self: pyaudi.abs(self)
gdual_double.sqrt = lambda self: pyaudi.sqrt(self)
gdual_double.exp = lambda self: pyaudi.exp(self)
gdual_double.expm1 = lambda self: pyaudi.exp(self) - 1.0
gdual_double.log = lambda self: pyaudi.log(self)
gdual_double.log10 = lambda self: pyaudi.log(self) / pyaudi.log(gdual_double(10.0))
gdual_double.log1p = lambda self: pyaudi.log(self + 1.0)
gdual_double.log2 = lambda self: pyaudi.log(self) / pyaudi.log(gdual_double(2.0))
gdual_double.cos = lambda self: pyaudi.cos(self)
gdual_double.sin = lambda self: pyaudi.sin(self)
gdual_double.tan = lambda self: pyaudi.tan(self)
gdual_double.cosh = lambda self: pyaudi.cosh(self)
gdual_double.sinh = lambda self: pyaudi.sinh(self)
gdual_double.tanh = lambda self: pyaudi.tanh(self)
gdual_double.arccos = lambda self: pyaudi.acos(self)
gdual_double.arcsin = lambda self: pyaudi.asin(self)
gdual_double.arctan = lambda self: pyaudi.atan(self)
gdual_double.arctan2 = lambda self, x2: __atan2_helper(self, x2)
gdual_double.erf = lambda self: pyaudi.erf(self)
gdual_double.erfc = lambda self: 1.0 - pyaudi.erf(self)

float_fmts.update({
    'gdual_double': gdual_double,
})

# gdual_vdouble Definitions
gdual_vdouble.__float__ = lambda self: self.constant_cf
# gdual_vdouble.__int__ = lambda self: int(self.constant_cf)
gdual_vdouble.__abs__ = lambda self: pyaudi.abs(self)
gdual_vdouble.sqrt = lambda self: pyaudi.sqrt(self)
gdual_vdouble.exp = lambda self: pyaudi.exp(self)
gdual_vdouble.expm1 = lambda self: pyaudi.exp(self) - 1.0
gdual_vdouble.log = lambda self: pyaudi.log(self)
gdual_vdouble.log10 = lambda self: pyaudi.log(self) / numpy.log(10.0)
gdual_vdouble.log1p = lambda self: pyaudi.log(self + 1.0)
gdual_vdouble.log2 = lambda self: pyaudi.log(self) / numpy.log(2.0)
gdual_vdouble.cos = lambda self: pyaudi.cos(self)
gdual_vdouble.sin = lambda self: pyaudi.sin(self)
gdual_vdouble.tan = lambda self: pyaudi.tan(self)
gdual_vdouble.cosh = lambda self: pyaudi.cosh(self)
gdual_vdouble.sinh = lambda self: pyaudi.sinh(self)
gdual_vdouble.tanh = lambda self: pyaudi.tanh(self)
gdual_vdouble.arccos = lambda self: pyaudi.acos(self)
gdual_vdouble.arcsin = lambda self: pyaudi.asin(self)
gdual_vdouble.arctan = lambda self: pyaudi.atan(self)
gdual_vdouble.arctan2 = lambda self, x2: __atan2_helper(self, x2)
gdual_vdouble.erf = lambda self: pyaudi.erf(self)
gdual_vdouble.erfc = lambda self: 1.0 - pyaudi.erf(self)

float_fmts.update({
    'gdual_vdouble': gdual_vdouble,
})


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
    return 1 / (1 + exp(-x1))


# Reduction Ops
def logsumexp(x1, *args, **kwargs):
    if x1.dtype == object:
        return log(sum(exp(x1)))
    else:
        return scipy.special.logsumexp(x1, *args, **kwargs)
    
def __matrix_inv_helper(A_in, tol=epsilon()):
    if shape(A_in)[0] != shape(A_in)[1]:
        raise numpy.linalg.LinAlgError("Matrix is not square, inversion is not possible")
    detA = numpy.linalg.det(to_float(A_in))
    if detA == 0:
        raise numpy.linalg.LinAlgError("Matrix has a determinant of {} which makes it non-invertible".format(detA))
    A = copy(A_in)
    I = eye(shape(A)[0], dtype=A.dtype)
    h = 0 # Initialization of the pivot row
    k = 0 # Initialization of the pivot column

    while h < shape(A)[0] and k < shape(A)[1]:
        # Find the k-th pivot: 
        i_max = h + argmax(abs(to_float(A[h:, k])))
        if to_float(A[i_max, k]) == 0:
            # No pivot in this column, pass to next column
            k = k + 1
        else:
            A[h], A[i_max] = copy(A[i_max]), copy(A[h])
            for i in range(h+1, shape(A)[0]):
                f = A[i, k] / A[h, k]
                A[i, :] = A[i, :] - f * A[h, :]
                I[i, :] = I[i, :] - f * I[h, :]
            # Increase pivot row and column
            h = h + 1
            k = k + 1
    for k in range(shape(A)[1]-1, -1, -1):
        I[k, :] = I[k, :] / A[k, k]
        A[k, :] = A[k, :] / A[k, k]
        f         = (A[k-1,k] / A[k,k])
        A[k-1, :] = A[k-1, :] - f*A[k,:]
        I[k-1, :] = I[k-1, :] - f*I[k, :]
        
    Ik = I@A_in
    for _ in range(10):
        I  = 2*I - Ik@I
        Ik = I@A_in
        if max(abs(to_float(Ik))) - 1 <= tol:
            break
    return I
    
def __matrix_inv_dispatcher(A, *args, **kwargs):
    if A.dtype == object:
        if len(shape(A)) == 2:
            return __matrix_inv_helper(A, *args, **kwargs)
        else:
            return reshape(stack([
                __matrix_inv_helper(A_batch, *args, **kwargs) for A_batch in reshape(A, (-1, *shape(A)[-2:]))
            ]), shape(A))
    else:
        return numpy.linalg.inv(A, *args, **kwargs)

matrix_inv = __matrix_inv_dispatcher

# gdual_real128 Definitions
# try:
#     gdual_real128 = pyaudi.gdual_real128
#     gdual_real128.__float__ = lambda self: float(repr(self.constant_cf))
#     #    gdual_real128.__int__     = lambda self: int(float(repr(self.constant_cf)))
#     gdual_real128.__abs__ = lambda self: pyaudi.abs(self)
#     gdual_real128.sqrt = lambda self: pyaudi.sqrt(self)
#     gdual_real128.exp = lambda self: pyaudi.exp(self)
#     gdual_real128.expm1 = lambda self: pyaudi.exp(self) - 1.0
#     gdual_real128.log = lambda self: pyaudi.log(self)
#     gdual_real128.log10 = lambda self: pyaudi.log(self) / pyaudi.log(gdual_real128(10.0))
#     gdual_real128.log1p = lambda self: pyaudi.log(self + 1.0)
#     gdual_real128.log2 = lambda self: pyaudi.log(self) / pyaudi.log(gdual_real128(2.0))
#     gdual_real128.cos = lambda self: pyaudi.cos(self)
#     gdual_real128.sin = lambda self: pyaudi.sin(self)
#     gdual_real128.tan = lambda self: pyaudi.tan(self)
#     gdual_real128.cosh = lambda self: pyaudi.cosh(self)
#     gdual_real128.sinh = lambda self: pyaudi.sinh(self)
#     gdual_real128.tanh = lambda self: pyaudi.tanh(self)
#     gdual_real128.arccos = lambda self: pyaudi.acos(self)
#     gdual_real128.arcsin = lambda self: pyaudi.asin(self)
#     gdual_real128.arctan = lambda self: pyaudi.atan(self)
#     gdual_real128.arctan2 = lambda self, x2: __atan2_helper(self, x2)
#     gdual_real128.erf = lambda self: pyaudi.erf(self)
#     gdual_real128.erfc = lambda self: 1.0 - pyaudi.erf(self)
#
#     float_fmts.update({
#         'gdual_real128': gdual_real128,
#     })
# except AttributeError:
#     pass
# except Exception as e:
#     print("Unable to load pyaudi.gdual_real128", file=sys.stderr)
#     print("\t\tError raised is:", file=sys.stderr)
#     print(e, file=sys.stderr)
#     pass
