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
        

@pytest.mark.parametrize('ffmt', D.available_float_fmt())        
def test_jacobian_wrapper_non_callable(ffmt):
    D.set_float_fmt(ffmt)
    rhs = 5.0
    
    jac_rhs = de.utilities.JacobianWrapper(rhs)
    
    with pytest.raises(TypeError):
        print(jac_rhs(0.1))
        

@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_jacobian_wrapper_exact(ffmt):
    D.set_float_fmt(ffmt)
    rhs     = lambda x: D.exp(-x)
    drhs_exact = lambda x: -D.exp(-x)
    jac_rhs = de.utilities.JacobianWrapper(rhs, rtol=D.epsilon() ** 0.5, atol=D.epsilon() ** 0.5)
    
    assert (D.allclose(drhs_exact(0.0), jac_rhs(0.0), rtol=4 * D.epsilon() ** 0.5, atol=4 * D.epsilon() ** 0.5))
        

@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_jacobian_wrapper_no_adaptive(ffmt):
    D.set_float_fmt(ffmt)
    rhs     = lambda x: D.exp(-x)
    drhs_exact = lambda x: -D.exp(-x)
    jac_rhs = de.utilities.JacobianWrapper(rhs, richardson_iter=4, adaptive=False, rtol=D.epsilon() ** 0.5, atol=D.epsilon() ** 0.5)
    
    assert (D.allclose(drhs_exact(0.0), jac_rhs(0.0), rtol=4 * D.epsilon() ** 0.5, atol=4 * D.epsilon() ** 0.5))
        

@pytest.mark.parametrize('ffmt', D.available_float_fmt())
def test_jacobian_wrapper_calls_estimate(ffmt):
    D.set_float_fmt(ffmt)
    rhs     = lambda x: D.exp(-x)
    jac_rhs = de.utilities.JacobianWrapper(rhs, richardson_iter=0, adaptive=False, rtol=D.epsilon() ** 0.5, atol=D.epsilon() ** 0.5)
    
    assert (D.allclose(jac_rhs.estimate(0.0), jac_rhs(0.0), rtol=4 * D.epsilon() ** 0.5, atol=4 * D.epsilon() ** 0.5))

    
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


if __name__ == "__main__":
    np.testing.run_module_suite()
