from desolver.backend.common import *

import torch
import autoray

linear_algebra_exceptions.append(torch._C._LinAlgError)


def __solve_linear_system(A:torch.Tensor, b:torch.Tensor, sparse=False):
    """Solves a linear system either exactly when A is invertible, or
    approximately when A is not invertible"""
    eps_threshold = torch.finfo(b.dtype).eps**0.5
    soln = torch.empty_like(A[...,0,:,None])
    is_square = A.shape[-2] == A.shape[-1]
    if is_square:
        use_solve = torch.linalg.det(A).abs() > eps_threshold
    else:
        use_solve = torch.zeros_like(soln[...,0,0], dtype=torch.bool)
    info = torch.ones_like(use_solve, dtype=torch.int)
    soln, info = torch.linalg.solve_ex(A, b, check_errors=False)
    use_solve = use_solve & ((info == 0) | torch.all(torch.isfinite(soln[...,0]), dim=-1))
    use_svd = ~use_solve
    U,S,Vh = torch.linalg.svd(A, full_matrices=is_square)
    if A.dim() == 2:
        soln = (Vh.mT @ torch.linalg.pinv(torch.diag_embed(S)) @ U.mT @ b)
    else:
        soln = torch.where(
            use_svd[...,None,None],
            torch.bmm(torch.bmm(torch.bmm(Vh.mT, torch.linalg.pinv(torch.diag_embed(S))), U.mT), b),
            soln,
        )
    return soln


def to_cpu_wrapper(fn):
    def new_fn(x:torch.Tensor):
        return fn(x.cpu())
    return new_fn


def detach_wrapper(fn):
    def new_fn(x:torch.Tensor):
        return fn(x.detach())
    return new_fn


def bfloat16_compat_wrapper(fn):
    def new_fn(x:torch.Tensor):
        if x.dtype == torch.bfloat16:
            return fn(x.to(torch.float32))
        else:
            return fn(x)
    return new_fn


def copyto(dst:torch.Tensor, src:torch.Tensor, casting=None, where:torch.Tensor|None=None):
    if where is not None:
        dst.masked_scatter_(where, src)
    else:
        dst.copy_(src)


def place(dst:torch.Tensor, mask:torch.Tensor, src:torch.Tensor):
    return dst.masked_scatter_(mask, src)


autoray.register_function("torch", "to_numpy", bfloat16_compat_wrapper, wrap=True)
autoray.register_function("torch", "to_numpy", to_cpu_wrapper, wrap=True)
autoray.register_function("torch", "to_numpy", detach_wrapper, wrap=True)
autoray.register_function("torch", "solve_linear_system", __solve_linear_system)
autoray.register_function("torch", "copyto", copyto)
autoray.register_function("torch", "place", place)
autoray.autoray._FUNC_ALIASES[('torch', 'copy')] = 'clone'
