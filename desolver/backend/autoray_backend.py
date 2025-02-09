from autoray import numpy as ar_numpy
from functools import lru_cache
import autoray
import numpy as np
import einops


@lru_cache(maxsize=32, typed=False)
def epsilon(dtype: str|np.dtype):
    if isinstance(dtype, str) and 'numpy' in dtype:
        dtype = np.dtype(dtype)
    if dtype in (np.half, np.single, np.double, np.longdouble):
        return np.finfo(dtype).eps*4
    elif 'torch' in str(dtype):
        import torch
        return torch.finfo(dtype).eps*4
    else:
        return 4e-14


@lru_cache(maxsize=32, typed=False)
def tol_epsilon(dtype: str|np.dtype):
    if isinstance(dtype, str) and 'numpy' in dtype:
        dtype = np.dtype(dtype)
    if dtype in (np.half, np.single, np.double, np.longdouble):
        return np.finfo(dtype).eps*32
    elif 'torch' in str(dtype):
        import torch
        return torch.finfo(dtype).eps*32
    else:
        return 32e-14


@lru_cache(maxsize=32, typed=False)
def backend_like_dtype(dtype: str|np.dtype):
    if (isinstance(dtype, str) and 'numpy' in dtype) or isinstance(dtype, np.dtype):
        return 'numpy'
    elif 'torch' in str(dtype):
        return 'torch'
    