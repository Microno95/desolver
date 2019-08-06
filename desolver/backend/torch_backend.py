from .common import *

import torch

# Datatypes
bool    = torch.uint8
float16 = torch.float16
float32 = torch.float32
float64 = torch.float64
uint8   = torch.uint8
int16   = torch.int16
int32   = torch.int32
int64   = torch.int64

def asarray(x):
    if not torch.is_tensor(x):
        return array(x)
    return x

def to_float(x):
    if not torch.is_tensor(x):
        return torch.tensor(x, dtype=torch.get_default_dtype())
    return x

def to_type(x, dtype):
    if not torch.is_tensor(x):
        return torch.tensor(x, dtype=dtype)
    return x.to(dtype)

# Convenience Decorators
def type_reg(f):
    def _wrapped(*args, **kwargs):
        kwargs.setdefault("dtype", torch.get_default_dtype())
        return f(*args, **kwargs)
    _wrapped.original_function = f
    return _wrapped

def axis_reg(f):
    def _wrapped(x, axis=None, *args, **kwargs):
        if axis is None:
            return f(x.view(-1), dim=0, *args, **kwargs)
        return f(x, dim=axis, *args, **kwargs)
    _wrapped.original_function = f
    return _wrapped

def keepdim_reg(f):
    def _wrapped(x, keepdims=False, *args, **kwargs):
        return f(x, keepdim=keepdims, *args, **kwargs)
    _wrapped.original_function = f
    return _wrapped

# Fundamental Mathematical Operators
neg      = torch.neg
pow      = torch.pow
abs      = torch.abs
sqrt     = torch.sqrt
rsqrt    = torch.rsqrt

exp      = torch.exp
expm1    = torch.expm1
log      = torch.log
log10    = torch.log10
log1p    = torch.log1p
log2     = torch.log2

add      = torch.add
mul      = torch.mul
div      = torch.div

addcdiv  = torch.addcdiv
addcmul  = torch.addcmul
reciprocal = torch.reciprocal
remainder  = torch.remainder

ceil     = torch.ceil
floor    = torch.floor
round    = torch.round
fmod     = torch.fmod
frac     = torch.frac
lerp     = torch.lerp
clip     = torch.clamp
sign     = torch.sign
trunc    = torch.trunc

# Trigonometric Functions
cos      = torch.cos
sin      = torch.sin
tan      = torch.tan

cosh     = torch.cosh
sinh     = torch.sinh
tanh     = torch.tanh 

acos     = torch.acos
asin     = torch.asin
atan     = torch.atan
atan2    = torch.atan2

# Other Functions
digamma  = torch.digamma
mvlgamma = torch.mvlgamma
erf      = torch.erf
erfc     = torch.erfc
erfinv   = torch.erfinv
sigmoid  = torch.sigmoid

# Additional Math Definitions
def square(x, out=None):
    if out is not None:
        out.data = x**2
        return out
    else:
        return x**2

def sub(x, y, out=None):
    if out is None:
        return x - y
    neg(y, out=out)
    return add(x, out, out=out)

# Common Array Operations
einsum      = torch.einsum
arange      = type_reg(torch.arange)

def concatenate(arrs, axis=0, out=None):
    if axis is None:
        return torch.cat([i.view(-1) for i in arrs], dim=0, out=out)
    return torch.cat(arrs, dim=axis, out=out)

def ravel(x):
    return asarray(x).flatten()

def flatten(x):
    return ravel(x)

def append(arr, values, axis=None):
    arr = asarray(arr)
    if axis is None:
        if arr.dim() != 1:
            arr = ravel(arr)
        values = ravel(values)
        axis = arr.dim()-1
    return concatenate((arr, values), axis=axis)

def stack(arrs, axis=0, out=None):
    return torch.stack(arrs, dim=axis, out=out)

@type_reg
def linspace(start, end, num=100, out=None, dtype=None):
    return torch.linspace(start, end, steps=num, dtype=dtype, out=out)

@type_reg
def logspace(start, end, num=100, out=None, dtype=None):
    return torch.logspace(start, end, steps=num, dtype=dtype, out=out)

@type_reg
def eye(N, M=None, out=None, dtype=None):
    if out is None:
        return torch.eye(N, m=M, dtype=dtype)
    else:
        out.data = torch.eye(N, m=M, dtype=dtype)
    return out

# Reduction Ops
argmax    = keepdim_reg(axis_reg(torch.argmax))
argmin    = keepdim_reg(axis_reg(torch.argmin))
cumprod   = keepdim_reg(axis_reg(torch.cumprod))
cumsum    = keepdim_reg(axis_reg(torch.cumsum))
logsumexp = keepdim_reg(axis_reg(torch.logsumexp))
mean      = keepdim_reg(axis_reg(torch.mean))
median    = keepdim_reg(axis_reg(torch.median))
prod      = keepdim_reg(axis_reg(torch.prod))
std       = keepdim_reg(axis_reg(torch.std))
var       = keepdim_reg(axis_reg(torch.var))
sum       = keepdim_reg(axis_reg(torch.sum))

def norm(x, ord='fro', axis=None, keepdims=False):
    return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)

def dist(x, y, ord=2):
    return torch.dist(x, y, p=ord)

# Comparison Ops
allclose   = torch.allclose
argsort    = axis_reg(torch.argsort)

eq         = torch.eq
ne         = torch.ne
ge         = torch.ge
gt         = torch.gt
le         = torch.le
lt         = torch.lt

equal      = torch.equal
isfinite   = torch.isfinite
isinf      = torch.isinf
isnan      = torch.isnan

def max(x, axis=None, keepdims=False, out=None):
    if axis is None:
        return torch.max(x.view(-1), dim=0, keepdim=keepdims, out=out)[0]
    return torch.max(x, dim=axis, keepdim=keepdims, out=out)[0]

def min(x, axis=None, keepdims=False, out=None):
    if axis is None:
        return torch.min(x.view(-1), dim=0, keepdim=keepdims, out=out)[0]
    return torch.min(x, dim=axis, keepdim=keepdims, out=out)[0]

array      = type_reg(torch.tensor)
zeros      = type_reg(torch.zeros)
empty      = type_reg(torch.empty)
full       = type_reg(torch.full)
zeros_like = type_reg(torch.zeros_like)
ones_like  = type_reg(torch.ones_like)
empty_like = type_reg(torch.empty_like)
full_like  = type_reg(torch.full_like)

def to_numpy(x):
    return x.detach().numpy()

def as_bool_array(x):
    return x.to(bool)

def copy(x):
    return x.clone()

def reshape(x, new_dims):
    return torch.reshape(x, new_dims)

def shape(x):
    return x.shape

def logical_not(a, out=None, where=None):
    if where is None:
        out = ~a
    else:
        if out is None:
            out = a.clone()
        out[where] = ~out[where]
    return out

def logical_or(a, b, out=None, where=None):
    if where is None:
        out = a | b
    else:
        if out is None:
            out = a.clone()
        out[where] = a[where] | b[where]
    return out

def logical_and(a, b, out=None, where=None):
    if where is None:
        out = a & b
    else:
        if out is None:
            out = a.clone()
        out[where] = a[where] & b[where]
    return out

def logical_xor(a, b, out=None, where=None):
    if where is None:
        out = a ^ b
    else:
        if out is None:
            out = a.clone()
        out[where] = a[where] ^ b[where]
    return out

def jacobian(inputs, outputs, batch_mode=False, nu=1, create_graph=True):
    # Computes the jacobian matrix of a given pytorch function/module
    # wrt the inputs.
    # Can be slow for higher order derivatives due to the recursion and
    # the exponential increase in the number of parameters.
    # (Could be optimized via taking advantage of symmetries in the resulting
    # tensors.)
    if batch_mode:
        outputs_view = outputs.view(outputs.shape[0], -1)
        temp = sum(
            torch.autograd.grad(
                outputs_view[:, j], 
                inputs,
                grad_outputs=torch.ones_like(outputs_view[:, j]),
                allow_unused=True,
                retain_graph=True,
                create_graph=create_graph if nu==1 else True
            )[0] for j in range(outputs_view.shape[1]))
        temp = torch.reshape(temp, outputs.shape + inputs.shape[1:])
    else:
        outputs_view = outputs.view(-1)
        temp = [torch.autograd.grad(
            outputs_view[i], 
            inputs,
            allow_unused=True,
            retain_graph=True,
            create_graph=create_graph if nu==1 else True,
        )[0] for i in range(outputs_view.shape[0])]
        temp = torch.stack([
            i if i is not None else torch.zeros_like(inputs) for i in temp
        ])
        if temp[0] is None:
            temp = torch.reshape(torch.stack([torch.zeros_like(inputs) for _ in outputs.numel()]), outputs.shape + inputs.shape)
        else:
            temp = torch.reshape(temp, outputs.shape + inputs.shape)
    if nu > 1:
        temp = jacobian(inputs, temp, create_graph=create_graph, nu=nu-1, batch_mode=batch_mode)
    return temp