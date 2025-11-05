import desolver as de
import desolver.backend as D
import numpy as np
import pytest
import copy
from desolver.tests import common


def test_getter_setters(dtype_var, backend_var, device_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    if backend_var == 'torch':
        arr_con_kwargs['device'] = device_var
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)

    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon(dtype_var) ** 0.5,
                     atol=D.epsilon(dtype_var) ** 0.5)

    assert (a.t0 == 0)
    assert (a.tf == 2 * D.pi)
    assert (a.dt == 0.01)
    assert (a.get_current_time() == a.t0)
    assert (a.rtol == D.epsilon(dtype_var) ** 0.5)
    assert (a.atol == D.epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.linalg.norm(a.y[0] - y_init) <= 2 * D.epsilon(dtype_var))
    assert (D.ar_numpy.linalg.norm(a.y[-1] - y_init) <= 2 * D.epsilon(dtype_var))

    a.set_kick_vars([True, False])

    assert (a.staggered_mask == [True, False])
    pval = 3 * D.pi

    a.tf = pval

    assert (a.tf == pval)
    pval = -1.0

    a.t0 = pval

    assert (a.t0 == pval)
    assert (a.dt == 0.01)

    a.rtol = 1e-3

    assert (a.rtol == 1e-3)

    a.atol = 1e-3

    assert (a.atol == 1e-3)

    for method in de.available_methods():
        a.set_method(method)
        assert (isinstance(a.integrator, de.available_methods(False)[method]))

    for method in de.available_methods():
        a.method = method
        assert (isinstance(a.integrator, de.available_methods(False)[method]))

    a.constants['k'] = 5.0

    assert (a.constants['k'] == 5.0)

    a.constants.pop('k')

    assert ('k' not in a.constants.keys())

    new_constants = dict(k=10.0)

    a.constants = new_constants

    assert (a.constants['k'] == 10.0)

    del a.constants

    assert (not bool(a.constants))


@common.integrator_param
def test_integration_and_representation_no_jac(dtype_var, backend_var, integrator):
    print()
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    
    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var, integrator=integrator)
    a.tf = D.pi/4

    assert (a.integration_status == "Integration has not been run.")
    
    if a.integrator.is_implicit and D.ar_numpy.finfo(dtype_var).bits < 32:
        pytest.skip(f"{a.integrator} is unstable for {D.ar_numpy.finfo(dtype_var).bits}-bit precision")
    elif a.integrator.order <= 6 and D.ar_numpy.finfo(dtype_var).bits > 32:
        pytest.skip(f"{a.integrator}'s order is too low for {D.ar_numpy.finfo(dtype_var).bits}-bit precision")
    elif a.integrator.is_implicit and D.ar_numpy.finfo(dtype_var).bits > 64:
        pytest.skip(f"{a.integrator}'s is too slow for {D.ar_numpy.finfo(dtype_var).bits}-bit precision")
    
    if D.ar_numpy.finfo(dtype_var).eps > 64:
        tol = a.atol = a.rtol = 1e-12
        test_tol = (tol*32)**0.5
    else:
        test_tol = D.tol_epsilon(dtype_var) ** 0.5
    if a.integrator.order <= 6:
        test_tol = 128 * test_tol
    if a.integrator.is_adaptive and a.integrator.order > 8:
        a.dt = a.dt * 0.01
    print(test_tol, a.atol, a.rtol)
    
    a.integrate(eta=True)

    assert (a.integration_status == "Integration completed successfully.")

    print(str(a))
    print(repr(a))
    try:
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[0])) - D.ar_numpy.to_numpy(y_init))) <= test_tol)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a.t[-1], y_init)))) <= test_tol)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t).T) - D.ar_numpy.to_numpy(analytic_soln(a.t, y_init)))) <= test_tol)

        for i in a:
            assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= test_tol)

        assert (len(a.y) == len(a))
        assert (len(a.t) == len(a))
        assert (a.success)
    except AssertionError as e:
        if backend_var == 'torch' and D.ar_numpy.finfo(dtype_var).bits < 32:
            pytest.xfail(f"Low precision {dtype_var} can fail some of the tests: {e}")
        elif backend_var == 'numpy' and D.ar_numpy.finfo(dtype_var).bits < 32:
            pytest.xfail(f"Low precision {dtype_var} can fail some of the tests: {e}")
        else:
            raise e


@common.implicit_integrator_param
def test_integration_and_representation_with_jac(dtype_var, backend_var, integrator):
    print()
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    
    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var, integrator=integrator, hook_jacobian=False)
    a.tf = D.pi/4

    assert (a.integration_status == "Integration has not been run.")
    
    if a.integrator.is_implicit and D.ar_numpy.finfo(dtype_var).bits < 32:
        pytest.skip(f"{a.integrator} is unstable for {D.ar_numpy.finfo(dtype_var).bits}-bit precision")
    elif a.integrator.order <= 6 and D.ar_numpy.finfo(dtype_var).bits > 32:
        pytest.skip(f"{a.integrator}'s order is too low for {D.ar_numpy.finfo(dtype_var).bits}-bit precision")
    elif a.integrator.is_implicit and D.ar_numpy.finfo(dtype_var).bits > 64:
        pytest.skip(f"{a.integrator}'s is too slow for {D.ar_numpy.finfo(dtype_var).bits}-bit precision")
    
    if D.ar_numpy.finfo(dtype_var).eps > 64:
        tol = a.atol = a.rtol = 1e-12
        test_tol = (tol*32)**0.5
    else:
        test_tol = D.tol_epsilon(dtype_var) ** 0.5
    if a.integrator.order <= 6:
        test_tol = 128 * test_tol
    if a.integrator.is_adaptive and a.integrator.order > 8:
        a.dt = a.dt * 0.01
    
    a.integrate(eta=True)

    assert (a.integration_status == "Integration completed successfully.")

    print(str(a))
    print(repr(a))
    try:
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[0])) - D.ar_numpy.to_numpy(y_init))) <= test_tol)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a.t[-1], y_init)))) <= test_tol)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t).T) - D.ar_numpy.to_numpy(analytic_soln(a.t, y_init)))) <= test_tol)

        for i in a:
            assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= test_tol)

        assert (len(a.y) == len(a))
        assert (len(a.t) == len(a))
        
        if backend_var == 'torch':
            # Test rehooking of jac through autodiff
            
            (de_mat, rhs, analytic_soln, y_init, dt, a_torch) = common.set_up_basic_system(dtype_var, backend_var, integrator=integrator, hook_jacobian=True)
            a_torch.tf = D.pi/4

            assert (a_torch.integration_status == "Integration has not been run.")

            a_torch.equ_rhs.unhook_jacobian_call()
            
            for i in a:
                assert (D.ar_numpy.max(D.ar_numpy.abs(a_torch.equ_rhs.jac(i.t, i.y) - a.equ_rhs.jac(i.t, i.y))) <= test_tol)
            
            if D.ar_numpy.finfo(dtype_var).eps > 64:
                tol = a_torch.atol = a_torch.rtol = 1e-12
                test_tol = (tol*32)**0.5
            else:
                test_tol = D.tol_epsilon(dtype_var) ** 0.5
            if a_torch.integrator.order <= 4:
                test_tol = 128 * test_tol
            
            a_torch.integrate(eta=True)

            assert (a_torch.integration_status == "Integration completed successfully.")

            print(str(a_torch))
            print(repr(a_torch))
            assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a_torch.sol(a_torch.t[0])) - D.ar_numpy.to_numpy(y_init))) <= test_tol)
            assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a_torch.sol(a_torch.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a_torch.t[-1], y_init)))) <= test_tol)
            assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a_torch.sol(a_torch.t).T) - D.ar_numpy.to_numpy(analytic_soln(a_torch.t, y_init)))) <= test_tol)

            for i in a_torch:
                assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= test_tol)

            assert (len(a_torch.y) == len(a_torch))
            assert (len(a_torch.t) == len(a_torch))
            assert (a_torch.nfev > 0)
            assert (a_torch.njev > 0)
    except AssertionError as e:
        if backend_var == 'torch' and D.ar_numpy.finfo(dtype_var).bits < 32:
            pytest.xfail(f"Low precision {dtype_var} can fail some of the tests: {e}")
        elif backend_var == 'numpy' and D.ar_numpy.finfo(dtype_var).bits < 32:
            pytest.xfail(f"Low precision {dtype_var} can fail some of the tests: {e}")
        else:
            raise e


@common.basic_explicit_integrator_param
def test_integration_with_richardson(dtype_var, backend_var, integrator):
    print()
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    
    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var, integrator=integrator)

    assert (a.integration_status == "Integration has not been run.")
    
    a.method = de.integrators.generate_richardson_integrator(a.method, richardson_iter=2 if D.ar_numpy.finfo(dtype_var).bits < 32 else 4)
    
    test_tol = D.tol_epsilon(dtype_var) ** 0.5
    a.integrate()

    assert (a.integration_status == "Integration completed successfully.")

    print(str(a))
    print(repr(a))
    try:
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[0])) - D.ar_numpy.to_numpy(y_init))) <= test_tol)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a.t[-1], y_init)))) <= test_tol)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t).T) - D.ar_numpy.to_numpy(analytic_soln(a.t, y_init)))) <= test_tol)

        for i in a:
            assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= test_tol)

        assert (len(a.y) == len(a))
        assert (len(a.t) == len(a))
    except AssertionError as e:
        if backend_var == 'torch' and D.ar_numpy.finfo(dtype_var).bits < 32:
            pytest.xfail(f"Low precision {dtype_var} can fail some of the tests: {e}")
        elif backend_var == 'numpy' and D.ar_numpy.finfo(dtype_var).bits < 32:
            pytest.xfail(f"Low precision {dtype_var} can fail some of the tests: {e}")
        else:
            raise e


def test_integration_and_nearest_float_no_dense_output(dtype_var, backend_var, device_var):
    print()
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
        
    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    if backend_var == 'torch':
        arr_con_kwargs['device'] = device_var
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)

    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon(dtype_var) ** 0.5,
                     atol=D.epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))

    assert (a.integration_status == "Integration has not been run.")

    a.integrate()
    
    assert (a.sol is None)

    assert (a.integration_status == "Integration completed successfully.")

    # assert (D.ar_numpy.abs(a.t[-2] - a[2 * D.pi].t) <= D.ar_numpy.abs(a.dt))

    assert (len(a.events) == 0)


def test_integration_reset(dtype_var, backend_var, device_var):
    print()
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
        
    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    if backend_var == 'torch':
        arr_con_kwargs['device'] = device_var
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)

    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon(dtype_var) ** 0.5,
                     atol=D.epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))

    assert (a.integration_status == "Integration has not been run.")

    a.integrate()
    
    assert (a.sol is None)

    assert (a.integration_status == "Integration completed successfully.")

    assert (len(a.events) == 0)

    pre_reset_a = copy.deepcopy(a)

    a.reset()
    a.integrate(eta=True)
    
    assert (a.sol is None)

    assert (a.integration_status == "Integration completed successfully.")

    assert (len(a.events) == 0)

    assert D.ar_numpy.allclose(pre_reset_a.t, a.t)
    assert D.ar_numpy.allclose(pre_reset_a.y, a.y)


def test_integration_long_duration(dtype_var, backend_var):
    print()
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    if D.ar_numpy.finfo(dtype_var).bits < 32:
        pytest.skip(f"dtype: {dtype_var} lacks the precision for long-duration integration")
        
    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)

    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 128 * D.pi), dt=0.01, rtol=D.epsilon(dtype_var) ** 0.5,
                     atol=D.epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))

    assert (a.integration_status == "Integration has not been run.")

    a.integrate(eta=True)
    
    assert (a.sol is None)

    assert (a.integration_status == "Integration completed successfully.")
    
    tol = D.epsilon(dtype_var)

    assert (D.ar_numpy.allclose(a.t[-1], 128*D.pi*D.ar_numpy.ones_like(a.t[-1]), tol, tol))

    assert (len(a.events) == 0)


def test_wrong_t0(dtype_var, backend_var):
    with pytest.raises(ValueError):
        dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
        if backend_var == 'torch':
            import torch
            torch.set_printoptions(precision=17)
            torch.autograd.set_detect_anomaly(True)

        arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
        de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
            
        @de.rhs_prettifier("""[vx, -x+t]""")
        def rhs(t, state, k, **kwargs):
            t = D.ar_numpy.atleast_1d(t)
            return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

        y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

        a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=0.01, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                         atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))

        a.t0 = 2 * D.pi


def test_wrong_tf(dtype_var, backend_var):
    with pytest.raises(ValueError):
        dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
        if backend_var == 'torch':
            import torch
            torch.set_printoptions(precision=17)
            torch.autograd.set_detect_anomaly(True)

        arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
        de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
            
        @de.rhs_prettifier("""[vx, -x+t]""")
        def rhs(t, state, k, **kwargs):
            t = D.ar_numpy.atleast_1d(t)
            return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

        y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

        a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=0.01, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                         atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))

        a.tf = 0.0


def test_not_enough_time_values(dtype_var, backend_var):
    with pytest.raises(ValueError):
        dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
        if backend_var == 'torch':
            import torch
            torch.set_printoptions(precision=17)
            torch.autograd.set_detect_anomaly(True)

        arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
        de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
            
        @de.rhs_prettifier("""[vx, -x+t]""")
        def rhs(t, state, k, **kwargs):
            t = D.ar_numpy.atleast_1d(t)
            return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

        y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

        a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0,), dt=0.01, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                         atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))


def test_dt_dir_fix(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
        
    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2*D.pi), dt=-0.5, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                        atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))
    
    assert a.dt == 0.5


def test_non_callable_rhs(dtype_var, backend_var):
    with pytest.raises(ValueError):
        dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
        if backend_var == 'torch':
            import torch
            torch.set_printoptions(precision=17)
            torch.autograd.set_detect_anomaly(True)

        arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
        de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
            
        @de.rhs_prettifier("""[vx, -x+t]""")
        def rhs(t, state, k, **kwargs):
            t = D.ar_numpy.atleast_1d(t)
            return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)

        y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

        a = de.OdeSystem(de_mat, y0=y_init, dense_output=False, t=(0,), dt=0.01, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                         atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))


def test_callback_called(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
        
    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)
    
    callback_called = False
    
    def callback(ode_sys):
        nonlocal callback_called
        if not callback_called and ode_sys.t[-1] > D.pi:
            callback_called = True

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2*D.pi), dt=-0.5, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                        atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))
    
    a.integrate(callback=callback)
    
    assert(callback_called)


def test_backward_integration(dtype_var, backend_var):
    print()
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    
    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var)
    a.tf = -2*D.pi

    assert (a.integration_status == "Integration has not been run.")
    
    if D.ar_numpy.finfo(dtype_var).eps > 64:
        tol = a.atol = a.rtol = 1e-12
        test_tol = (tol*32)**0.5
    else:
        test_tol = D.tol_epsilon(dtype_var) ** 0.5
    if a.integrator.order <= 6:
        test_tol = 128 * test_tol
    
    a.integrate(eta=True)

    assert (a.integration_status == "Integration completed successfully.")

    print(str(a))
    print(repr(a))
    try:
        assert (a.t[-1] < a.t[0])
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[0])) - D.ar_numpy.to_numpy(y_init))) <= test_tol)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a.t[-1], y_init)))) <= test_tol)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t).T) - D.ar_numpy.to_numpy(analytic_soln(a.t, y_init)))) <= test_tol)

        for i in a:
            assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= test_tol)

        assert (len(a.y) == len(a))
        assert (len(a.t) == len(a))
        assert (a.success)
    except AssertionError as e:
        if backend_var == 'torch' and D.ar_numpy.finfo(dtype_var).bits < 32:
            pytest.xfail(f"Low precision {dtype_var} can fail some of the tests: {e}")
        elif backend_var == 'numpy' and D.ar_numpy.finfo(dtype_var).bits < 32:
            pytest.xfail(f"Low precision {dtype_var} can fail some of the tests: {e}")
        else:
            raise e


@common.basic_integrator_param
@pytest.mark.parametrize("datatype", ["float32", "float64"])
def test_mixed_environment(integrator, datatype):
    torch = pytest.importorskip("torch")
    
    np_oscillator_mat    = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=D.autoray.to_backend_dtype(datatype, like='numpy'))
    torch_oscillator_mat = torch.tensor(np_oscillator_mat, dtype=D.autoray.to_backend_dtype(datatype, like='torch'))
        
    def np_rhs(t, state):
        t = np.atleast_1d(t)
        return np_oscillator_mat @ state + np.concatenate([np.zeros_like(t), t], axis=0) - 0.001*state**2

    def torch_rhs(t, state):
        t = torch.atleast_1d(t)
        return torch_oscillator_mat @ state + torch.cat([torch.zeros_like(t), t], dim=0) - 0.001*state**2
    
    t_span = [0.0, 10.0]
    np_y0 = np.array([0.0, 1.0], dtype=D.autoray.to_backend_dtype(datatype, like='numpy'))
    torch_y0 = torch.tensor(np_y0, dtype=D.autoray.to_backend_dtype(datatype, like='torch'))
    atol = rtol = 512*D.tol_epsilon(D.autoray.to_backend_dtype(datatype, like='numpy'))**0.5
    
    ode_sys_numpy = de.OdeSystem(np_rhs, y0=np_y0, dense_output=False, t=t_span, dt=0.001, atol=atol, rtol=rtol)
    ode_sys_numpy.set_kick_vars([False, True])
    ode_sys_numpy.method = integrator
    ode_sys_torch = de.OdeSystem(torch_rhs, y0=torch_y0, dense_output=False, t=t_span, dt=0.001, atol=atol, rtol=rtol)
    ode_sys_numpy.set_kick_vars([False, True])
    ode_sys_torch.method = integrator
    
    ode_sys_numpy.integrate(eta=True)
    print(repr(ode_sys_numpy))
    
    ode_sys_torch.integrate(eta=True)
    print(repr(ode_sys_torch))
    
    assert np.allclose(ode_sys_numpy.y[0], ode_sys_torch.y[0].numpy(), rtol, atol)
    assert np.allclose(ode_sys_numpy.y[-1], ode_sys_torch.y[-1].numpy(), rtol, atol)


def test_keyboard_interrupt_caught(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
        
    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)
    
    def kb_callback(ode_sys):
        if ode_sys.t[-1] > D.pi:
            raise KeyboardInterrupt()

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2*D.pi), dt=-0.5, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                        atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))

    with pytest.raises(KeyboardInterrupt):
        a.integrate(callback=kb_callback)
    
    assert(a.integration_status == "A KeyboardInterrupt exception was raised during integration.")


def test_equ_repr_attribute(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
        
    # @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)
    
    rhs.equ_repr = """[vx, -x+t]"""
    rhs.md_repr  = """$[v, -x+t]$"""

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2*D.pi), dt=-0.5, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                        atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))
        
    assert a.equ_rhs.equ_repr == rhs.equ_repr
    assert a.equ_rhs.md_repr == rhs.md_repr


def test_not_callable_rhs(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
        
    # @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)
    
    rhs.equ_repr = """[vx, -x+t]"""
    rhs.md_repr  = """$[v, -x+t]$"""

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)

    with pytest.raises(TypeError):
        a = de.OdeSystem(None, y0=y_init, dense_output=False, t=(0, 2*D.pi), dt=-0.5, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                            atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))


def test_incompatible_shape(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    arr_con_kwargs = dict(dtype=dtype_var, like=backend_var)
    de_mat = D.ar_numpy.asarray([[0.0, 1.0], [-1.0, 0.0]], **arr_con_kwargs)
        
    # @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, k, **kwargs):
        t = D.ar_numpy.atleast_1d(t)
        return de_mat @ state + D.ar_numpy.concatenate([D.ar_numpy.zeros_like(t), t], axis=0)
    
    rhs.equ_repr = """[vx, -x+t]"""
    rhs.md_repr  = """$[v, -x+t]$"""

    y_init = D.ar_numpy.asarray([1., 0.], **arr_con_kwargs)[None]

    with pytest.raises(RuntimeError if backend_var == "torch" else ValueError):
        a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2*D.pi), dt=-0.5, rtol=D.tol_epsilon(dtype_var) ** 0.5,
                            atol=D.tol_epsilon(dtype_var) ** 0.5, constants=dict(k=1.0))
    

def test_DiffRHS():
    def rhs(t, state, k, **kwargs):
        return None

    wrapped_rhs_no_repr = de.DiffRHS(rhs)

    assert (str(wrapped_rhs_no_repr) == str(rhs))
    assert (wrapped_rhs_no_repr._repr_markdown_() == str(rhs))

    wrapped_rhs_no_equ_repr = de.DiffRHS(rhs, equ_repr=None, md_repr="1")

    assert (str(wrapped_rhs_no_equ_repr) == str(rhs))
    assert (wrapped_rhs_no_equ_repr._repr_markdown_() == "1")

    wrapped_rhs_no_md_repr = de.DiffRHS(rhs, equ_repr="1", md_repr=None)

    assert (str(wrapped_rhs_no_md_repr) == "1")
    assert (wrapped_rhs_no_md_repr._repr_markdown_() == "1")

    wrapped_rhs_both_repr = de.DiffRHS(rhs, equ_repr="1", md_repr="2")

    assert (str(wrapped_rhs_both_repr) == "1")
    assert (wrapped_rhs_both_repr._repr_markdown_() == "2")
    
    rhs_matrix = np.array([
        [0.0, 1.0],
        [-1.0, 0.01]
    ], dtype=np.float64)
    
    def rhs(t, state, **kwargs):
        return rhs_matrix @ state
    
    jac_called = False
    
    def jac(t, state, **kwargs):
        nonlocal jac_called
        jac_called = True
        return rhs_matrix

    x0 = np.array([[0.5, 0.5]], dtype=np.float64).mT

    wrapped_rhs_no_jac = de.DiffRHS(rhs)
    
    assert (D.ar_numpy.allclose(rhs_matrix @ x0, wrapped_rhs_no_jac(0.0, x0), D.epsilon(np.float64)**0.5))
    assert (D.ar_numpy.allclose(rhs_matrix, wrapped_rhs_no_jac.jac(0.0, x0)[:,0,:,0], D.epsilon(np.float64)**0.5))
    
    wrapped_rhs_with_jac = de.DiffRHS(rhs)
    wrapped_rhs_with_jac.hook_jacobian_call(jac)
    
    assert (D.ar_numpy.allclose(rhs_matrix @ x0, wrapped_rhs_with_jac(0.0, x0), D.epsilon(np.float64)**0.5))
    assert (D.ar_numpy.allclose(rhs_matrix, wrapped_rhs_with_jac.jac(0.0, x0), D.epsilon(np.float64)**0.5))
    assert (jac_called)


@pytest.mark.parametrize('integrator', [(de.integrators.RK45CKSolver, 'RK45'), (de.integrators.RadauIIA5, 'Radau'), (de.integrators.RK8713MSolver, 'LSODA')])
def test_solve_ivp_parity(integrator):
    print()
    from scipy.integrate import solve_ivp
    
    de_mat = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        
    def fun(t, state):
        t = np.atleast_1d(t)
        return de_mat @ state + np.concatenate([np.zeros_like(t), t], axis=0) - 0.001*state**2
    
    t_span = [0.0, 10.0]
    y0 = np.array([0.0, 1.0], dtype=np.float64)
    atol = rtol = 1e-10
    
    desolver_res = de.solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, method=integrator[0])
    scipy_res = solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, method=integrator[1])
    
    print(desolver_res)
    print(scipy_res)
    print(D.ar_numpy.mean(D.ar_numpy.diff(desolver_res.t)), D.ar_numpy.mean(D.ar_numpy.diff(scipy_res.t)))
    test_tol = 1e-6
    
    print(scipy_res.t[0] - desolver_res.t[0])
    assert np.allclose(scipy_res.t[0], desolver_res.t[0], test_tol, test_tol)
    print(scipy_res.t[-1] - desolver_res.t[-1])
    assert np.allclose(scipy_res.t[-1], desolver_res.t[-1], test_tol, test_tol)
    print(scipy_res.y[...,0] - desolver_res.y[...,0])
    assert np.allclose(scipy_res.y[...,0], desolver_res.y[...,0], test_tol, test_tol)
    print(scipy_res.y[...,-1] - desolver_res.y[...,-1])
    assert np.allclose(scipy_res.y[...,-1], desolver_res.y[...,-1], test_tol, test_tol)
    
    t_eval = np.linspace(*t_span, 32)
    
    desolver_res = de.solve_ivp(fun, t_span=t_span, t_eval=t_eval, y0=y0, atol=atol, rtol=rtol, method=integrator[0])
    scipy_res = solve_ivp(fun, t_span=t_span, t_eval=t_eval, y0=y0, atol=atol, rtol=rtol, method=integrator[1])
    
    print(desolver_res)
    print(scipy_res)
    print(D.ar_numpy.mean(D.ar_numpy.diff(desolver_res.t)), D.ar_numpy.mean(D.ar_numpy.diff(scipy_res.t)))
    test_tol = 1e-6
    
    print(scipy_res.t - desolver_res.t)
    assert np.allclose(scipy_res.t, desolver_res.t, test_tol, test_tol)
    print(scipy_res.y - desolver_res.y)
    assert np.allclose(scipy_res.y, desolver_res.y, test_tol, test_tol)
        
    def fun(t, state, k, m):
        de_mat = np.array([[0.0, 1.0], [-k/m, 0.0]], dtype=np.float64)
        t = np.atleast_1d(t)
        return de_mat @ state + np.concatenate([np.zeros_like(t), t], axis=0) - 0.001*state**2

    desolver_res = de.solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, method=integrator[0], args=(4.0, 0.1))
    scipy_res = solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, method=integrator[1], args=(4.0, 0.1))
    
    print(desolver_res)
    print(scipy_res)
    print(D.ar_numpy.mean(D.ar_numpy.diff(desolver_res.t)), D.ar_numpy.mean(D.ar_numpy.diff(scipy_res.t)))
    test_tol = 1e-6
    
    print(scipy_res.t[0] - desolver_res.t[0])
    assert np.allclose(scipy_res.t[0], desolver_res.t[0], test_tol, test_tol)
    print(scipy_res.t[-1] - desolver_res.t[-1])
    assert np.allclose(scipy_res.t[-1], desolver_res.t[-1], test_tol, test_tol)
    print(scipy_res.y[...,0] - desolver_res.y[...,0])
    assert np.allclose(scipy_res.y[...,0], desolver_res.y[...,0], test_tol, test_tol)
    print(scipy_res.y[...,-1] - desolver_res.y[...,-1])
    assert np.allclose(scipy_res.y[...,-1], desolver_res.y[...,-1], test_tol, test_tol)

    desolver_res = de.solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, min_step=1e-3, method=integrator[0], args=(4.0, 0.1))
    assert np.diff(desolver_res.t)[:-1].min() >= 1e-3 - 1e-8

    desolver_res = de.solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, max_step=1e-2, method=integrator[0], args=(4.0, 0.1))
    assert np.diff(desolver_res.t)[:-1].max() <= 1e-2 + 1e-8

    
    with pytest.raises(ValueError):
        t_eval = np.array([-1.0, 0.0, 10.0])
        desolver_res = de.solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, t_eval=t_eval, method=integrator[0], args=(4.0, 0.1))
    
    with pytest.raises(ValueError):
        t_eval = np.array([0.0, 10.0, 11.0])
        desolver_res = de.solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, t_eval=t_eval, method=integrator[0], args=(4.0, 0.1))


@pytest.mark.parametrize('integrator', [de.integrators.RK108Solver, de.integrators.RK8713MSolver, de.integrators.RadauIIA5, de.integrators.LobattoIIIC4, de.integrators.RadauIIA19])
def test_solve_stiff_system(integrator, backend_var):
    print()

    dtype_var = D.autoray.to_backend_dtype("float64", like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    
    @de.DiffRHS
    def fun(t, state):
        return -2000*(state - np.cos(t))
    
    def fun_jac(t, state):
        return D.ar_numpy.array([[-2000]], dtype=dtype_var, like=backend_var)
    
    fun.hook_jacobian_call(fun_jac)

    def solution(t):
        return D.ar_numpy.exp(-2000*t)/4000001 + (2000/4000001)*D.ar_numpy.sin(t) + (4000000/4000001)*D.ar_numpy.cos(t)

    t_span = [0.0, 5.0]
    y0 = D.ar_numpy.array([1.0], dtype=dtype_var, like=backend_var)
    atol = rtol = 1e-6

    desolver_res = de.solve_ivp(fun, t_span=t_span, y0=y0, atol=atol, rtol=rtol, method=integrator, show_prog_bar=True)
    
    print(desolver_res)
    print(D.ar_numpy.mean(D.ar_numpy.diff(desolver_res.t)))
    print(D.ar_numpy.mean(D.ar_numpy.abs(desolver_res.y - solution(desolver_res.t))))
    test_tol = atol**0.5
    
    assert D.ar_numpy.allclose(desolver_res.y, solution(desolver_res.t), test_tol, test_tol)