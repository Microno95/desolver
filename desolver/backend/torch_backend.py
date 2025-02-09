from desolver.backend.common import *

import torch
import autoray


def __solve_linear_system(A, b, sparse=False):
    __A = A
    __b = b
    if __A.dtype in (torch.float16, torch.bfloat16):
        __A = __A.float()
    if __b.dtype in (torch.float16, torch.bfloat16):
        __b = __b.float()
    return torch.linalg.solve(__A, __b).to(A.dtype)


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


autoray.register_function("torch", "to_numpy", bfloat16_compat_wrapper, wrap=True)
autoray.register_function("torch", "to_numpy", to_cpu_wrapper, wrap=True)
autoray.register_function("torch", "to_numpy", detach_wrapper, wrap=True)
autoray.register_function("torch", "solve_linear_system", __solve_linear_system)
autoray.autoray._FUNC_ALIASES[('torch', 'copy')] = 'clone'
