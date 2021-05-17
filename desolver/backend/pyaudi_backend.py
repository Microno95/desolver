import numpy
import pyaudi
import scipy.linalg
import scipy.sparse
import scipy.special
import builtins

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
    if numpy.asanyarray(x).dtype == object:
        __shape = numpy.asanyarray(x).shape
        __ndim = len(__shape)
        if __ndim > 0:
            unstacked = [to_float(i) for i in numpy.asanyarray(x).ravel()]
            max_dim = builtins.max(builtins.map(shape, unstacked), key=len)
            __shape_perm = list(range(__ndim + len(max_dim)))
            if len(max_dim) > 0:
                __shape_perm = [__shape_perm[-1], *__shape_perm[:-1]]
            ret = numpy.stack([numpy.resize(i, max_dim) for i in unstacked], axis=0).reshape(*__shape, *max_dim).astype(float64)
            return ret.transpose(__shape_perm)
        else:
            if isinstance(numpy.asanyarray(x).item(), gdual_vdouble):
                return numpy.array(numpy.asanyarray(x).item().constant_cf, dtype=float64)
            else:
                return numpy.array(float(numpy.asanyarray(x).item())).astype(float64)
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
    if shape(A_in)[-2] != shape(A_in)[-1]:
        raise numpy.linalg.LinAlgError("Matrix is not square, inversion is not possible")
    if len(shape(A_in)) > 2:
        return reshape(stack([
            __matrix_inv_helper(__A) for __A in reshape(A_in, (-1, shape(A_in)[-2], shape(A_in)[-1]))
        ]), shape(A_in))
    floatA = to_float(A_in)
    detA = stack([
        numpy.linalg.det(iA) for iA in reshape(floatA, (-1, *shape(floatA)[-2:]))
    ]).reshape(*shape(floatA)[:-2], -1)
    if any(detA == 0):
        raise numpy.linalg.LinAlgError("Matrix has a determinant of {} which makes it non-invertible".format(detA))
    A = copy(A_in)
    I = eye(shape(A)[0], dtype=A.dtype)
    P = eye(shape(A)[0], dtype=A.dtype)
    h = 0 # Initialization of the pivot row
    k = 0 # Initialization of the pivot column

    while h < shape(A)[0] and k < shape(A)[1]:
        # Find the k-th pivot: 
        i_max = h + argmax(abs(to_float(A[h:, k])))
        if to_float(A[i_max, k]) == 0:
            # No pivot in this column, pass to next column
            k = k + 1
        else:
            A[h], A[i_max] = A[i_max], copy(A[h])
            P[h], P[i_max] = P[i_max], copy(P[h])
            f = A[h+1:, k] / A[h, k]
            A[h+1:, :] = A[h+1:, :] - f[:,None] * A[h, :][None,:]
            I[h+1:, :] = I[h+1:, :] - f[:,None] * I[h, :][None,:]
            # Increase pivot row and column
            h = h + 1
            k = k + 1
    for k in range(shape(A)[1]-1, -1, -1):
        I[k, :] = I[k, :] / A[k, k]
        A[k, :] = A[k, :] / A[k, k]
        f       = (A[:k,k] / A[k,k])
        A[:k, :] = A[:k, :] - f[:,None]*A[k,:][None,:]
        I[:k, :] = I[:k, :] - f[:,None]*I[k, :][None,:]
        
    I = I@P
    Ident = eye(shape(A)[0], dtype=A.dtype)
    Ik = I@A_in
    for ref in range(30):
        AI = A_in@I
        Wn = -147*Ident + AI@(53*Ident + AI@(-11*Ident + AI))
        I = 0.25*I@(32*Ident + AI@(-113*Ident + AI@(231*Ident + AI@(-301*Ident + AI@(259*Ident + AI@Wn)))))
#         I  = 2*I - Ik@I
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

def __qr_gramschmidt(A):
    Q = zeros_like(A)
    for idx in range(len(A)):
        Q[:, idx] = A[:, idx]
        for jdx in range(idx):
            Q[:, idx] = Q[:, idx] - (Q[:, jdx].T@Q[:, idx])*Q[:, jdx]
        Q[:, idx] = Q[:, idx] / (Q[:, idx:idx+1].T@Q[:, idx:idx+1])**0.5
    R = Q.T@A
    return Q,R

def __qr_householder(A_in, overwrite_a=False):
    n = A_in.shape[0]
    if n == 1:
        return 1, A_in, A_in
    if overwrite_a:
        A = A_in
    else:
        A = copy(A_in)
    I = eye(n, dtype=A.dtype)
    Q = copy(I)
    for idx in range(n):
        v  = A[idx:, idx][:, None]
        dA = (v.T@v)**0.5
        if float(A[idx, idx]) < 0:
            v[0] = v[0] - dA
        else:
            v[0] = v[0] + dA         
        vn = (v.T@v)**0.5
        if float(vn) > 0.0:
            v = v / vn
        H = I[idx:, idx:] - 2*v@v.T
        A[idx:, :] = H@A[idx:, :]
        Q[:, idx:] = Q[:, idx:]@H
    return Q,A

def __qr_householder_vdouble(A_in, overwrite_a=False):
    n = A_in.shape[0]
    if n == 1:
        return 1, A_in, A_in
    if overwrite_a:
        A = A_in
    else:
        A = copy(A_in)
    I = eye(n, dtype=A.dtype)
    Q = copy(I)
    for idx in range(n):
        v  = A[idx:, idx][:, None]
        dA = (v.T@v)**0.5
        neg_mask = (to_float(A[idx, idx]) < 0).astype(float64).tolist()
        if not isinstance(neg_mask, list):
            neg_mask = [neg_mask]
        neg_mask = gdual_vdouble(neg_mask)
        corr     = -dA * neg_mask + dA * (1 - neg_mask)            
        v[0] = v[0] + corr
        vn = (v.T@v)**0.5
        pos_mask = gdual_vdouble((to_float(vn[0,0]) > 0.0).astype(float64).tolist())
        v = v / (vn * pos_mask + (1 - pos_mask))
        H = I[idx:, idx:] - 2*v@v.T
        A[idx:, :] = H@A[idx:, :]
        Q[:, idx:] = Q[:, idx:]@H
    return Q,A

def qr(A, overwrite_a=False):
    if len(A) < 20:
        return __qr_gramschmidt(A)
    else:
        if float_fmt() == 'gdual_vdouble':
            return __qr_householder_vdouble(A, overwrite_a=overwrite_a)
        else:
            return __qr_householder(A, overwrite_a=overwrite_a)

def __backward_substitution(U, b, overwrite_b=False):
    if overwrite_b:
        for idx in range(shape(U)[0]-1,-1,-1):
            b[idx] = b[idx] - sum(U[idx, idx+1:]*b[idx+1:, 0])
            b[idx] = b[idx] / U[idx, idx]
        return b
    else:
        x = copy(b)
        for idx in range(shape(U)[0]-1,-1,-1):
            x[idx] = x[idx] - sum(U[idx, idx+1:]*x[idx+1:, 0])
            x[idx] = x[idx] / U[idx, idx]
        return x

def __forward_substitution(L, b, overwrite_b=False):
    if overwrite_b:
        for idx in range(shape(L)[0]):
            b[idx] = b[idx] - sum(L[idx, :idx]*b[:idx, 0])
            b[idx] = b[idx] / L[idx, idx]
        return b
    else:
        x = copy(b)
        for idx in range(shape(L)[0]):
            x[idx] = x[idx] - sum(L[idx, :idx]*x[:idx, 0])
            x[idx] = x[idx] / L[idx, idx]
        return x

def __solve_linear_system_helper(A, b, overwrite_a=False):
    if shape(A)[0] < 30 or float_fmt() == "gdual_vdouble":
        if float_fmt() == "gdual_vdouble":
            Q,R = qr(A)
        else:
            Q,R = scipy.linalg.qr(to_float(A), overwrite_a=overwrite_a)
        bhat = Q.T@b
        y = __backward_substitution(R, bhat, overwrite_b=False)
        residual = bhat - R@y
        for _ in range(5):
            if max(abs(to_float(residual))) > 2*epsilon():
                y = y + __backward_substitution(R, residual, overwrite_b=False)
                residual = bhat - R@y
            else:
                break
        return y
    else:
        P,L,U = scipy.linalg.lu(to_float(A), overwrite_a=overwrite_a)
        y = P@b
        y = __forward_substitution(L, y,overwrite_b=True)
        return __backward_substitution(U, y, overwrite_b=True)
    
def __solve_linear_system_dispatcher(A, b, overwrite_a=False, overwrite_b=False, check_finite=False, sparse=False, *args, **kwargs):
    if A.dtype == object or b.dtype == object:
        if len(shape(A)) == 2:
            return __solve_linear_system_helper(A, b, overwrite_a=overwrite_a, *args, **kwargs)
        else:
            assert (shape(A)[-2] == shape(b)[-2]), "Batched system must have compatible shapes for A and b!"
            return reshape(stack([
                __solve_linear_system_helper(A_batch, b_batch, overwrite_a=overwrite_a, *args, **kwargs) for A_batch, b_batch in zip(reshape(A, (-1, *shape(A)[-2:])), reshape(b, (-1, *shape(b)[-2:])))
            ]), shape(b))
    else:
        if sparse:
            return scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A), b)
        else:
            return scipy.linalg.solve(A, b, overwrite_a=overwrite_a, overwrite_b=overwrite_b, check_finite=check_finite)
    
solve_linear_system = __solve_linear_system_dispatcher

def eval_weights(x, weights):
    if isinstance(x, (gdual_double, gdual_vdouble)):
        return x.evaluate(weights)
    elif isinstance(x, numpy.ndarray):
        return reshape(array(list(map(lambda y: eval_weights(y, weights), ravel(x)))), x.shape)
    else:
        return x
        
        