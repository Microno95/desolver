import desolver as de
from desolver.differential_system import DenseOutput
from desolver.utilities.interpolation import CubicHermiteInterp
import desolver.backend as D
import numpy as np
import pytest


def test_dense_init_and_call():
    denseoutput = DenseOutput(None, None)
    assert (denseoutput.t_eval == [0.0])
    assert (denseoutput.y_interpolants == [])


def test_dense_init_add_postfix():
    denseoutput = DenseOutput(None, None)
    inputs = D.array([0, 1, 0, 1, 1, 1])
    interpolator = CubicHermiteInterp(*inputs)
    denseoutput.add_interpolant(1, interpolator)


def test_dense_init_add_prefix():
    denseoutput = DenseOutput(None, None)
    inputs = D.array([-1, 0, 0, 1, 1, 1])
    interpolator = CubicHermiteInterp(*inputs)
    denseoutput.add_interpolant(-1, interpolator)


def test_dense_add_noncallable():
    with pytest.raises(TypeError):
        inputs = D.array([0, 1, 0, 1, 1, 1])
        interpolator = CubicHermiteInterp(*inputs)
        denseoutput = DenseOutput([0, 1], [interpolator])
        denseoutput.add_interpolant(2, None)


def test_dense_add_outofbounds():
    with pytest.raises(ValueError):
        inputs = D.array([0, 1, 0, 1, 1, 1])
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
        inputs = D.array([0, 1, 0, 1, 1, 1])
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


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_dense_output(ffmt):
    D.set_float_fmt(ffmt)

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(True)

    print("Testing {} float format".format(D.float_fmt()))

    from . import common

    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system()

    assert (a.integration_status() == "Integration has not been run.")

    a.integrate()

    assert (a.integration_status() == "Integration completed successfully.")

    assert (D.max(D.abs(a[0].y - analytic_soln(a[0].t, y_init))) <= 4 * D.epsilon())
    assert (D.max(D.abs(a[0].t)) <= 4 * D.epsilon())
    assert (D.max(D.abs(a[-1].y - analytic_soln(a[-1].t, y_init))) <= 10 * D.epsilon() ** 0.5)

    assert (D.max(D.abs(a[a[0].t].y - analytic_soln(a[0].t, y_init))) <= 4 * D.epsilon())
    assert (D.max(D.abs(a[a[0].t].t)) <= 4 * D.epsilon())
    assert (D.max(D.abs(a[a[-1].t].y - analytic_soln(a[-1].t, y_init))) <= 10 * D.epsilon() ** 0.5)

    assert (D.max(D.abs(D.stack(a[a[0].t:a[-1].t].y) - D.stack(a.y))) <= 4 * D.epsilon())
    assert (D.max(D.abs(D.stack(a[:a[-1].t].y) - D.stack(a.y))) <= 4 * D.epsilon())

    assert (D.max(D.abs(D.stack(a[a[0].t:a[-1].t:2].y) - D.stack(a.y[::2]))) <= 4 * D.epsilon())
    assert (D.max(D.abs(D.stack(a[a[0].t::2].y) - D.stack(a.y[::2]))) <= 4 * D.epsilon())
    assert (D.max(D.abs(D.stack(a[:a[-1].t:2].y) - D.stack(a.y[::2]))) <= 4 * D.epsilon())


if __name__ == "__main__":
    np.testing.run_module_suite()
