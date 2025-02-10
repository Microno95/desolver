import desolver as de
from desolver.utilities.interpolation import CubicHermiteInterp
import desolver.backend as D
import numpy as np
import pytest


def test_cubic_interpolation(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    t0, t1 = D.ar_numpy.asarray([0.0, 2.0], dtype=dtype_var, like=backend_var)
    p0, p1 = D.ar_numpy.asarray([0.0, 2.0], dtype=dtype_var, like=backend_var)
    m0, m1 = D.ar_numpy.asarray([1.0, 1.0], dtype=dtype_var, like=backend_var)
    test_interp = CubicHermiteInterp(t0, t1, p0, p1, m0, m1)
    
    assert (D.ar_numpy.allclose(test_interp.tshift, t0))
    assert (D.ar_numpy.allclose(test_interp.trange, (t1 - t0)))
    
    test_t = D.ar_numpy.asarray([-1.0, 0.0, 1.0], dtype=dtype_var, like=backend_var)
    
    assert (D.ar_numpy.allclose(test_interp(t0), p0))
    assert (D.ar_numpy.allclose(test_interp.grad(t0), m0))
    assert (D.ar_numpy.allclose(test_interp(t1), p1))
    assert (D.ar_numpy.allclose(test_interp.grad(t1), m1))
    
    tmid = (t0 + t1)/2
    
    assert (D.ar_numpy.allclose(test_interp(tmid), (p0 + p1)/2))
    assert (D.ar_numpy.allclose(test_interp.grad(tmid), m1))
    
    assert (repr(test_interp) != "")
