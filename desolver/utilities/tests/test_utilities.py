import desolver as de
import desolver.backend as D
import numpy as np
import pytest


def test_convert_suffix():
    assert (de.utilities.convert_suffix(3661) == "0d:1h:1m1.00s")


def test_bisection_search():
    l1 = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
    for idx, i in enumerate(l1):
        assert (de.utilities.search_bisection(l1, i) == idx)


def test_bisection_search_vec():
    l1 = [1.0, 2.0, 3.0, 5.0, 10.0]
    l2 = [0.2, 1.2, 2.2, 3.2, 5.2, 9.2]
    expected_l2 = [0, 1, 2, 3, 4, 4]
    l3 = [0.9, 1.9, 2.9, 3.9, 5.9, 9.9]
    expected_l3 = [0, 1, 2, 3, 4, 4]
    l4 = [0.2, 9.9, 2.3, 5.5]
    expected_l4 = [0, 4, 2, 4]
    
    assert (D.ar_numpy.all(de.utilities.search_bisection_vec(l1, l2) == D.ar_numpy.asarray(expected_l2)))
    assert (D.ar_numpy.all(de.utilities.search_bisection_vec(l1, l3) == D.ar_numpy.asarray(expected_l3)))
    assert (D.ar_numpy.all(de.utilities.search_bisection_vec(l1, l4) == D.ar_numpy.asarray(expected_l4)))
        

def test_jacobian_wrapper_non_callable(dtype_var, backend_var, device_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    rhs = D.ar_numpy.asarray(5.0, dtype=dtype_var, like=backend_var)
    
    jac_rhs = de.utilities.JacobianWrapper(rhs)

    x = D.ar_numpy.asarray(0.1, dtype=dtype_var, like=backend_var)
    if backend_var == "torch":
        x = x.to(device_var)
    
    with pytest.raises(TypeError):
        print(jac_rhs(x))
        

def test_jacobian_wrapper_exact(dtype_var, backend_var, device_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    rhs     = lambda x: D.ar_numpy.exp(-x)
    drhs_exact = lambda x: -D.ar_numpy.exp(-x)
    
    x = D.ar_numpy.asarray(0.0, dtype=dtype_var, like=backend_var)
    if backend_var == "torch":
        x = x.to(device_var)
    
    # Adaptive estimate
    jac_rhs = de.utilities.JacobianWrapper(rhs, rtol=D.epsilon(dtype_var) ** 0.5, atol=D.epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.allclose(D.ar_numpy.to_numpy(drhs_exact(x)), D.ar_numpy.to_numpy(jac_rhs(x)), rtol=D.tol_epsilon(dtype_var) ** 0.5, atol=D.tol_epsilon(dtype_var) ** 0.5))
    
    # Non-adaptive estimate
    jac_rhs = de.utilities.JacobianWrapper(rhs, richardson_iter=4, adaptive=False, rtol=D.epsilon(dtype_var) ** 0.5, atol=D.epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.allclose(D.ar_numpy.to_numpy(drhs_exact(x)), D.ar_numpy.to_numpy(jac_rhs(x)), rtol=D.tol_epsilon(dtype_var) ** 0.5, atol=D.tol_epsilon(dtype_var) ** 0.5))
    
    # Non-adaptive estimate
    jac_rhs = de.utilities.JacobianWrapper(rhs, richardson_iter=0, adaptive=False, rtol=D.epsilon(dtype_var) ** 0.5, atol=D.epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.allclose(D.ar_numpy.to_numpy(jac_rhs.estimate(x)), D.ar_numpy.to_numpy(jac_rhs(x)), rtol=D.tol_epsilon(dtype_var) ** 0.5, atol=D.tol_epsilon(dtype_var) ** 0.5))

    
def test_blocktimer():
    with de.utilities.BlockTimer(start_now=False) as test:
        assert (test.start_time is None and not test.start_now)

    with de.utilities.BlockTimer(start_now=False) as test:
        assert (isinstance(test.start_now, bool) and test.start_now == False)
        assert (test.start_time is None)
        assert (test.end_time is None)
        test.start()
        assert (isinstance(test.start_time, float))
        assert (isinstance(test.elapsed(), float) and test.elapsed() > 0)
        test.end()
        assert (isinstance(test.end_time, float))
        assert (isinstance(test.elapsed(), float) and test.elapsed() > 0)
        assert (test.stopped == True)
        test.restart_timer()
        assert (test.end_time is None)
        assert (test.stopped == False)
