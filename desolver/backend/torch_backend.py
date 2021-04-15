from .common import *

import torch

# Datatypes
bool = torch.bool
float32 = torch.float32
float64 = torch.float64
uint8 = torch.uint8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64

float_fmts.update({
    'float32': float32,
    'float64': float64
})


def to_float(x):
    if not torch.is_tensor(x):
        return torch.tensor(x, dtype=torch.get_default_dtype())
    return x.to(torch.get_default_dtype()).clone().detach()


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
neg = torch.neg
pow = torch.pow
abs = torch.abs
sqrt = torch.sqrt
rsqrt = torch.rsqrt

exp = torch.exp
expm1 = torch.expm1
log = torch.log
log10 = torch.log10
log1p = torch.log1p
log2 = torch.log2

add = torch.add
mul = torch.mul
div = torch.div

addcdiv = torch.addcdiv
addcmul = torch.addcmul
reciprocal = torch.reciprocal
remainder = torch.remainder

ceil = torch.ceil
floor = torch.floor
round = torch.round
fmod = torch.fmod
frac = torch.frac
lerp = torch.lerp
clip = torch.clamp
sign = torch.sign
trunc = torch.trunc

# Trigonometric Functions
cos = torch.cos
sin = torch.sin
tan = torch.tan

cosh = torch.cosh
sinh = torch.sinh
tanh = torch.tanh

acos = torch.acos
asin = torch.asin
atan = torch.atan
atan2 = torch.atan2

# Other Functions
digamma = torch.digamma
mvlgamma = torch.mvlgamma
erf = torch.erf
erfc = torch.erfc
erfinv = torch.erfinv
sigmoid = torch.sigmoid


def softplus(x, out=None):
    if out is not None:
        exp(x, out=out)
        log(1 + out, out=out)
        return out
    else:
        return log(1 + exp(x))


# Additional Math Definitions
def square(x, out=None):
    if out is not None:
        pow(x, 2, out=out)
        return out
    else:
        return x ** 2


def sub(x, y, out=None):
    if out is None:
        return x - y
    neg(y, out=out)
    return add(x, out, out=out)


# Common Array Operations
einsum = torch.einsum
arange = type_reg(torch.arange)


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
        axis = arr.dim() - 1
    return concatenate((arr, values), axis=axis)


def stack(arrs, axis=0, out=None):
    return torch.stack(arrs, dim=axis, out=out)


@type_reg
def linspace(start, end, num=50, out=None, dtype=None):
    return torch.linspace(start, end, steps=num, dtype=dtype, out=out)


@type_reg
def logspace(start, end, num=50, out=None, dtype=None):
    return torch.logspace(start, end, steps=num, dtype=dtype, out=out)


@type_reg
def eye(N, M=None, out=None, dtype=None, device=None):
    if out is None:
        if M is None:
            return torch.eye(N, dtype=dtype, device=device)
        else:
            return torch.eye(N, m=M, dtype=dtype, device=device)
    else:
        if M is None:
            torch.eye(N, dtype=dtype, out=out, device=device)
        else:
            torch.eye(N, m=M, dtype=dtype, out=out, device=device)
        return out


# Reduction Ops
argmax = keepdim_reg(axis_reg(torch.argmax))
argmin = keepdim_reg(axis_reg(torch.argmin))
cumprod = keepdim_reg(axis_reg(torch.cumprod))
cumsum = keepdim_reg(axis_reg(torch.cumsum))
logsumexp = keepdim_reg(axis_reg(torch.logsumexp))
mean = keepdim_reg(axis_reg(torch.mean))
median = keepdim_reg(axis_reg(torch.median))
prod = keepdim_reg(axis_reg(torch.prod))
std = keepdim_reg(axis_reg(torch.std))
var = keepdim_reg(axis_reg(torch.var))
sum = keepdim_reg(axis_reg(torch.sum))


def norm(x, ord='fro', axis=None, keepdims=False):
    return torch.norm(x, p=ord, dim=axis, keepdim=keepdims)


def dist(x, y, ord=2):
    return torch.dist(x, y, p=ord)


# Comparison Ops
allclose = torch.allclose
argsort = axis_reg(torch.argsort)

eq = torch.eq
ne = torch.ne
ge = torch.ge
gt = torch.gt
le = torch.le
lt = torch.lt

equal = torch.equal
isfinite = torch.isfinite
isinf = torch.isinf
isnan = torch.isnan


def max(x, axis=None, keepdims=False, out=None):
    if axis is None:
        return torch.max(x.reshape(-1), dim=0, keepdim=keepdims, out=out)[0]
    return torch.max(x, dim=axis, keepdim=keepdims, out=out)[0]


def min(x, axis=None, keepdims=False, out=None):
    if axis is None:
        return torch.min(x.view(-1), dim=0, keepdim=keepdims, out=out)[0]
    return torch.min(x, dim=axis, keepdim=keepdims, out=out)[0]


any = torch.any
all = torch.all

@type_reg
def array(x, dtype=None, device=None, requires_grad=False):
    if not torch.is_tensor(x):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    out = x.detach().clone().to(dtype=dtype, device=device)
    out.requires_grad = requires_grad
    return out

# array = type_reg(torch.tensor)
zeros = type_reg(torch.zeros)
ones  = type_reg(torch.ones)
empty = type_reg(torch.empty)
full = type_reg(torch.full)
zeros_like = type_reg(torch.zeros_like)
ones_like = type_reg(torch.ones_like)
empty_like = type_reg(torch.empty_like)
full_like = type_reg(torch.full_like)


def asarray(x):
    if not torch.is_tensor(x):
        return array(x)
    return x


def to_numpy(x):
    if isinstance(x, (list, tuple)):
        return stack(x).detach().cpu().numpy()
    return x.clone().detach().cpu().numpy()


def as_bool_array(x):
    return x.to(bool)


def copy(x):
    return x.clone()


def reshape(x, new_dims):
    return torch.reshape(x, new_dims)


def shape(x):
    if hasattr(x, 'shape'):
        return x.shape
    else:
        return shape(asarray(x))


def logical_not(a, out=None, where=True):
    if where is None or (isinstance(where, type(True)) and where is True):
        if out is not None:
            torch.logical_not(a, out=out)
        else:
            out = torch.logical_not(a)
    else:
        if out is None:
            out = torch.zeros_like(a, dtype=torch.bool)
        out[where] = torch.logical_not(a[where])
    return out


def logical_or(a, b, out=None, where=True):
    if where is None or (isinstance(where, type(True)) and where is True):
        if out is not None:
            torch.logical_or(a, b, out=out)
        else:
            out = torch.logical_or(a, b)
    else:
        if out is None:
            out = torch.zeros_like(a, dtype=torch.bool)
        out[where] = a[where] | b[where]
    return out


def logical_and(a, b, out=None, where=True):
    if where is None or (isinstance(where, type(True)) and where is True):
        if out is not None:
            torch.logical_and(a, b, out=out)
        else:
            out = torch.logical_and(a, b)
    else:
        if out is None:
            out = torch.zeros_like(a, dtype=torch.bool)
        out[where] = a[where] & b[where]
    return out


def logical_xor(a, b, out=None, where=True):
    if where is None or (isinstance(where, type(True)) and where is True):
        if out is not None:
            torch.logical_xor(a, b, out=out)
        else:
            out = torch.logical_xor(a, b)
    else:
        if out is None:
            out = torch.zeros_like(a, dtype=torch.bool)
        out[where] = a[where] ^ b[where]
    return out


def nonzero(a):
    if len(shape(a)) == 0:
        return (torch.nonzero(a.reshape(-1)),)
    else:
        return (torch.nonzero(a),)


argsort = axis_reg(torch.argsort)

def gather(arr, indices, axis=0):
    return torch.gather(arr, dim=axis, index=indices)


def jacobian(out_tensor, in_tensor, batch_mode=False, nu=1, create_graph=True, return_intermediate=False):
    """Computes the derivative of an output tensor wrt an input tensor.

    Computes the full nu-th order derivative for the output tensor wrt an input tensor. 
    For nu = 1, this is the Jacobian, for nu = 2, this is the Hessian, etc.
    The computation scales with the number of output values, ie. out_tensor.numel(), thus it will
    become quite slow for very large tensors.
    
    The batched computation assumes that the first dimension is the batch dimension and computes the 
    derivative for all the batch elements. The batches are computed in parallel thus for reasonable batch
    sizes the computation should scale as out_tensor.numel() / out_tensor.shape[0].

    Parameters
    ----------
    out_tensor : torch.tensor
        The function whose derivative is to be computed
    in_tensor  : torch.tensor
        The input wrt which the derivative is to be computed
    batch_mode : bool
        Determines if the first dimension is to be treated as a batch dimension or not
    nu : int
        Order of the derivative to be computed
    create_graph : bool
        To keep the computational graph after the jacobian is computed. This is useful if you intend to
        compute further derivatives on the derivative, e.g. for gradient descent.

    Returns
    -------
    torch.tensor
        The derivative tensor of out_tensor wrt in_tensor

    Raises
    ------
    ValueError
        If nu < 0 as that is not a valid derivative order.

    See Also
    --------
    torch.autograd.grad : The base function through which gradients are computed

    Examples
    --------
    ```python
    >>> b   = torch.tensor( [0.0, 1.0], dtype=torch.float64, requires_grad=True)
    >>> mat = torch.tensor([[0.0, 1.0], [-5.0, 0.0]], dtype=torch.float64, requires_grad=True)
    >>> k   = mat@b
    >>> jacobian(k, b, nu=1)
    tensor([[ 0.,  1.],
            [-5.,  0.]], dtype=torch.float64)
    >>> jacobian(k, mat, nu=1)
    tensor([[[0., 1.],
             [0., 0.]],

            [[0., 0.],
             [0., 1.]]], dtype=torch.float64, grad_fn=<AsStridedBackward>)
    ```
    """
    if nu < 0:
        raise ValueError("nu cannot be less than zero! That's not a derivative...")
    if nu == 0:
        return out_tensor
    if out_tensor.requires_grad == False:
        if batch_mode:
            temp = torch.zeros(out_tensor.shape + in_tensor.shape[1:], dtype=in_tensor.dtype, device=out_tensor.device,
                               requires_grad=False)
        else:
            temp = torch.zeros(out_tensor.shape + in_tensor.shape, dtype=in_tensor.dtype, device=out_tensor.device,
                               requires_grad=False)
    else:
        if batch_mode:
            outputs_view = out_tensor.view(out_tensor.shape[0], -1)
            batch_one = torch.ones_like(outputs_view[:, 0])
            temp = [
                torch.autograd.grad(
                    outputs_view[:, j],
                    in_tensor,
                    grad_outputs=batch_one,
                    allow_unused=True,
                    retain_graph=True,
                    create_graph=create_graph if nu == 1 else True
                )[0] for j in range(outputs_view.shape[1])]
            final_shape = out_tensor.shape + in_tensor.shape[1:]
        else:
            outputs_view = out_tensor.view(-1)
            temp = [torch.autograd.grad(
                outputs_view[i],
                in_tensor,
                allow_unused=True,
                retain_graph=True,
                create_graph=create_graph if nu == 1 else True,
            )[0] for i in range(outputs_view.shape[0])]
            final_shape = out_tensor.shape + in_tensor.shape
        temp = torch.stack([
            i if i is not None else torch.zeros_like(in_tensor) for i in temp
        ])
        temp = temp.view(final_shape)
    if nu > 1:
        temp2 = jacobian(temp, in_tensor, create_graph=create_graph, nu=nu - 1, batch_mode=batch_mode, return_intermediate=return_intermediate)
        if return_intermediate:
            temp = (*temp2, (nu, temp))
        else:
            temp = temp2
    else:
        if return_intermediate:
            temp = ((nu, temp),)
    return temp

def solve_linear_system(A,b,sparse=False):
    return torch.solve(b,A).solution

matrix_inv = torch.linalg.inv
eig = torch.eig
diag = torch.diag