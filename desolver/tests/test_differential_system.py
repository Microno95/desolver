import desolver as de
import desolver.backend as D
import numpy as np
import pytest
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
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    
    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var, integrator=integrator)

    assert (a.integration_status == "Integration has not been run.")
    
    a.integrate()

    assert (a.integration_status == "Integration completed successfully.")

    print(str(a))
    print(repr(a))
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[0])) - D.ar_numpy.to_numpy(y_init))) <= D.tol_epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a.t[-1], y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t).T) - D.ar_numpy.to_numpy(analytic_soln(a.t, y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)

    for i in a:
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)

    assert (len(a.y) == len(a))
    assert (len(a.t) == len(a))


@common.implicit_integrator_param
def test_integration_and_representation_with_jac(dtype_var, backend_var, integrator):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    
    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var, integrator=integrator, hook_jacobian=True)

    assert (a.integration_status == "Integration has not been run.")

    a.integrate()

    assert (a.integration_status == "Integration completed successfully.")

    print(str(a))
    print(repr(a))
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[0])) - D.ar_numpy.to_numpy(y_init))) <= D.tol_epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a.t[-1], y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t).T) - D.ar_numpy.to_numpy(analytic_soln(a.t, y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)

    for i in a:
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)

    assert (len(a.y) == len(a))
    assert (len(a.t) == len(a))
    
    if backend_var == 'torch':
        # Test rehooking of jac through autodiff
        
        (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var, integrator=integrator, hook_jacobian=True)

        assert (a.integration_status == "Integration has not been run.")

        a.equ_rhs.unhook_jacobian_call()
        a.integrate()

        assert (a.integration_status == "Integration completed successfully.")

        print(str(a))
        print(repr(a))
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[0])) - D.ar_numpy.to_numpy(y_init))) <= D.tol_epsilon(dtype_var) ** 0.5)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a.t[-1], y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t).T) - D.ar_numpy.to_numpy(analytic_soln(a.t, y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)

        for i in a:
            assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)

        assert (len(a.y) == len(a))
        assert (len(a.t) == len(a))


@common.basic_explicit_integrator_param
def test_integration_with_richardson(dtype_var, backend_var, integrator):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
    
    (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system(dtype_var, backend_var, integrator=integrator)

    assert (a.integration_status == "Integration has not been run.")
    
    a.method = de.integrators.generate_richardson_integrator(a.method, richardson_iter=4)
    
    a.integrate()

    assert (a.integration_status == "Integration completed successfully.")

    print(str(a))
    print(repr(a))
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[0])) - D.ar_numpy.to_numpy(y_init))) <= D.tol_epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t[-1])) - D.ar_numpy.to_numpy(analytic_soln(a.t[-1], y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)
    assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(a.sol(a.t).T) - D.ar_numpy.to_numpy(analytic_soln(a.t, y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)

    for i in a:
        assert (D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(i.y) - D.ar_numpy.to_numpy(analytic_soln(i.t, y_init)))) <= D.tol_epsilon(dtype_var) ** 0.5)

    assert (len(a.y) == len(a))
    assert (len(a.t) == len(a))


def test_integration_and_nearest_float_no_dense_output(dtype_var, backend_var, device_var):
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

    assert (D.ar_numpy.abs(a.t[-2] - a[2 * D.pi].t) <= D.ar_numpy.abs(a.dt))

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
