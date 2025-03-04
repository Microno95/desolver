import desolver as de
from desolver.differential_system import DenseOutput
from desolver.utilities.interpolation import CubicHermiteInterp
import desolver.backend as D
import numpy as np
import pytest
from desolver.tests import common


def test_dense_init_and_call():
    denseoutput = DenseOutput(None, None)
    assert (denseoutput.t_eval == None)
    assert (denseoutput.y_interpolants == [])


def test_dense_init_add_postfix():
    denseoutput = DenseOutput(None, None)
    inputs = D.ar_numpy.asarray([0, 1, 0, 1, 1, 1])
    interpolator = CubicHermiteInterp(*inputs)
    denseoutput.add_interpolant(1, interpolator)


def test_dense_init_add_prefix():
    denseoutput = DenseOutput(None, None)
    inputs = D.ar_numpy.asarray([-1, 0, 0, 1, 1, 1])
    interpolator = CubicHermiteInterp(*inputs)
    denseoutput.add_interpolant(-1, interpolator)


def test_dense_add_noncallable():
    with pytest.raises(TypeError):
        inputs = D.ar_numpy.asarray([0, 1, 0, 1, 1, 1])
        interpolator = CubicHermiteInterp(*inputs)
        denseoutput = DenseOutput([0, 1], [interpolator])
        denseoutput.add_interpolant(2, None)


def test_dense_add_outofbounds():
    with pytest.raises(ValueError):
        inputs = D.ar_numpy.asarray([0, 1, 0, 1, 1, 1])
        interpolator = CubicHermiteInterp(*inputs)
        denseoutput = DenseOutput([0, 1], [interpolator])

        def new_interp(t):
            if t < 2:
                raise ValueError("Out of bounds")
            else:
                return t

        denseoutput.add_interpolant(2, new_interp)


def test_dense_add_timemismatch_oob():
    with pytest.raises(ValueError):
        inputs = D.ar_numpy.asarray([0, 1, 0, 1, 1, 1])
        interpolator = CubicHermiteInterp(*inputs)
        denseoutput = DenseOutput([0, 1], [interpolator])

        def new_interp(t):
            if t > 2:
                raise ValueError("Out of bounds")
            else:
                return t

        denseoutput.add_interpolant(3, new_interp)


def test_dense_init_no_t():
    with pytest.raises(ValueError):
        return DenseOutput(None, [0.1])


def test_dense_init_no_y():
    with pytest.raises(ValueError):
        return DenseOutput([0.1], None)


def test_dense_init_mismatch_length():
    with pytest.raises(ValueError):
        return DenseOutput([0.1], [0.1, 0.1])


def test_dense_right_interval(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    denseoutput = DenseOutput(None, None)
    inputs = D.ar_numpy.asarray([0, 1, 0, 1, 1, 1], dtype=dtype_var, like=backend_var)
    interpolator = CubicHermiteInterp(*inputs)
    denseoutput.add_interpolant(D.ar_numpy.asarray(1, dtype=dtype_var, like=backend_var), interpolator)
    denseoutput.add_interpolant(D.ar_numpy.asarray(2, dtype=dtype_var, like=backend_var), interpolator)
    assert (denseoutput.find_interval(0.5) == 0)
    assert (denseoutput.find_interval(1-D.tol_epsilon(dtype_var)) == 0)
    assert (denseoutput.find_interval(1+D.tol_epsilon(dtype_var)) == 1)
    assert (denseoutput.find_interval(1.5) == 1)


def test_dense_right_interval_vec(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    denseoutput = DenseOutput(None, None)
    inputs = D.ar_numpy.asarray([0, 1, 0, 1, 1, 1], dtype=dtype_var, like=backend_var)
    interpolator = CubicHermiteInterp(*inputs)
    denseoutput.add_interpolant(D.ar_numpy.asarray(1, dtype=dtype_var, like=backend_var), interpolator)
    denseoutput.add_interpolant(D.ar_numpy.asarray(2, dtype=dtype_var, like=backend_var), interpolator)
    test_t = D.ar_numpy.asarray([0.5, 1.0-D.tol_epsilon(dtype_var), 1+D.tol_epsilon(dtype_var), 1.5], dtype=dtype_var, like=backend_var)
    assert (D.ar_numpy.all(denseoutput.find_interval_vec(test_t) == D.ar_numpy.asarray([0, 0, 1, 1], dtype=D.autoray.to_backend_dtype("int64", like=backend_var), like=backend_var)))


@common.richardson_param
def test_dense_output(dtype_var, backend_var, use_richardson_extrapolation):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var)

    assert (a.integration_status == "Integration has not been run.")

    if use_richardson_extrapolation:
        a.method = de.integrators.generate_richardson_integrator(a.method)
    
    a.rtol = a.atol = D.tol_epsilon(dtype_var) ** 0.8
    a.integrate(eta=False)

    assert (a.integration_status == "Integration completed successfully.")

    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a[0].y) - D.ar_numpy.to_numpy(analytic_soln(a[0].t, y_init)))) <= D.tol_epsilon(dtype_var))
    assert (D.ar_numpy.max(D.ar_numpy.abs(a[0].t)) <= D.tol_epsilon(dtype_var))
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a[-1].y) - D.ar_numpy.to_numpy(analytic_soln(a[-1].t, y_init)))) <= 10 * D.tol_epsilon(dtype_var) ** 0.5)

    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a[a[0].t].y) - D.ar_numpy.to_numpy(analytic_soln(a[0].t, y_init)))) <= D.tol_epsilon(dtype_var))
    assert (D.ar_numpy.max(D.ar_numpy.abs(a[a[0].t].t)) <= D.tol_epsilon(dtype_var))
    assert (D.ar_numpy.max(
        D.ar_numpy.abs(D.ar_numpy.to_numpy(a[a[-1].t].y) - D.ar_numpy.to_numpy(analytic_soln(a[-1].t, y_init)))) <= 10 * D.tol_epsilon(dtype_var) ** 0.5)

    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a[a[0].t:a[-1].t].y) - D.ar_numpy.to_numpy(a.y))) <= D.tol_epsilon(dtype_var))
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a[:a[-1].t].y) - D.ar_numpy.to_numpy(a.y))) <= D.tol_epsilon(dtype_var))

    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a[a[0].t:a[-1].t:2].y) - D.ar_numpy.to_numpy(a.y[::2]))) <= D.tol_epsilon(dtype_var))
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a[a[0].t::2].y) - D.ar_numpy.to_numpy(a.y[::2]))) <= D.tol_epsilon(dtype_var))
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a[:a[-1].t:2].y) - D.ar_numpy.to_numpy(a.y[::2]))) <= D.tol_epsilon(dtype_var))

    if "float16" not in str(dtype_var):
        np.random.seed(42)
        sample_points = D.ar_numpy.asarray(np.random.uniform(a.t[0], a.t[-1], 8), dtype=dtype_var, like=backend_var)
        assert D.ar_numpy.allclose(a.sol(sample_points), analytic_soln(sample_points, y_init).mT, D.tol_epsilon(dtype_var) ** 0.5, D.tol_epsilon(dtype_var) ** 0.5)
