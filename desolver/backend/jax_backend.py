from desolver.backend.common import *

import jax
import jax.numpy
import autoray
import contextlib

@contextlib.contextmanager
def __no_grad_ctx():
    yield


def __solve_linear_system(A,b,overwrite_a=False,overwrite_b=False,check_finite=False,sparse=False):
    return jax.numpy.linalg.lstsq(A, b)[0]


autoray.register_function("jax", "solve_linear_system", __solve_linear_system)
autoray.register_function("jax", "no_grad", __no_grad_ctx)
autoray.register_function("jax", "clone", jax.numpy.copy)