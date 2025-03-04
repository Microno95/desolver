from desolver.backend.common import *

import numpy
import scipy
import scipy.special
import scipy.sparse
import scipy.sparse.linalg
import autoray


def __solve_linear_system(A,b,overwrite_a=False,overwrite_b=False,check_finite=False,sparse=False):
    if sparse and A.dtype not in (numpy.half, numpy.longdouble) and b.dtype not in (numpy.half, numpy.longdouble):
        return scipy.sparse.linalg.spsolve(scipy.sparse.csc_matrix(A),b)
    else:
        return scipy.linalg.solve(A,b,overwrite_a=overwrite_a,overwrite_b=overwrite_b,check_finite=check_finite)


autoray.register_function("numpy", "solve_linear_system", __solve_linear_system)
