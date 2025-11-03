from autoray import numpy as ar_numpy
from functools import lru_cache
import autoray
import numpy as np
import einops


@lru_cache(maxsize=32, typed=False)
def epsilon(dtype: str|np.dtype):
    if isinstance(dtype, str) and 'torch' not in dtype:
        dtype = np.dtype(dtype)
    try:
        return np.finfo(dtype).eps*4
    except:
        if 'torch' in str(dtype):
            import torch
            return torch.finfo(dtype).eps*4
        else:
            return 4e-14


@lru_cache(maxsize=32, typed=False)
def tol_epsilon(dtype: str|np.dtype):
    return 8*epsilon(dtype)


@lru_cache(maxsize=32, typed=False)
def backend_like_dtype(dtype: str|np.dtype):
    if 'torch' in str(dtype):
        return 'torch'
    else:
        return 'numpy'
    