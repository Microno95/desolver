import desolver as de
import desolver.backend as D
import pytest


integrator_set = set(de.available_methods(False).values())
integrator_set = sorted(integrator_set, key=lambda x: x.__name__)
explicit_integrator_set = [
    pytest.param(intg) for intg in integrator_set if intg((1,), D.numpy.float64).is_explicit
]
implicit_integrator_set = [
    pytest.param(intg, marks=pytest.mark.slow) for intg in integrator_set if intg((1,), D.numpy.float64).is_implicit
]

dt_set            = [D.pi / 307, D.pi / 512]
symplectic_integrator_set      = set(intg for intg in integrator_set if intg((1,), D.numpy.float64).symplectic)
integrator_param          = pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
explicit_integrator_param = pytest.mark.parametrize('integrator', explicit_integrator_set)
implicit_integrator_param = pytest.mark.parametrize('integrator', implicit_integrator_set)
basic_integrator_param    = pytest.mark.parametrize('integrator', [de.integrators.RK45CKSolver, de.integrators.RadauIIA5, de.integrators.ABAs5o6HSolver])
basic_explicit_integrator_param    = pytest.mark.parametrize('integrator', [de.integrators.RK45CKSolver, de.integrators.ABAs5o6HSolver])
richardson_param          = pytest.mark.parametrize('use_richardson_extrapolation', [False, True])
dt_param                  = pytest.mark.parametrize('dt', dt_set)
dense_output_param        = pytest.mark.parametrize('dense_output', [True, False])


def set_up_basic_system(dtype_var, backend_var, integrator=None, hook_jacobian=False):
    de_mat = D.ar_numpy.array([[0.0, 1.0], [-1.0, 0.0]], dtype=dtype_var, like=backend_var)

    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, **kwargs):
        nonlocal de_mat
        if D.autoray.infer_backend(state) == 'torch':
            de_mat = de_mat.to(state.device)
        out = de_mat @ state
        out[1] += t
        return out
    
    if hook_jacobian:
        def rhs_jac(t, state, **kwargs):
            nonlocal de_mat
            rhs.analytic_jacobian_called = True
            if D.autoray.infer_backend(state) == 'torch':
                de_mat = de_mat.to(state.device)
            return de_mat

        rhs.hook_jacobian_call(rhs_jac)

    def analytic_soln(t, initial_conditions):
        c1 = initial_conditions[0]
        c2 = initial_conditions[1] - 1.0

        t = D.ar_numpy.asarray(t, dtype=dtype_var, like=backend_var)
        
        return D.ar_numpy.stack([
            c2 * D.ar_numpy.sin(t) + c1 * D.ar_numpy.cos(t) + t,
            c2 * D.ar_numpy.cos(t) - c1 * D.ar_numpy.sin(t) + 1
        ])

    y_init = D.ar_numpy.array([1., 0.], dtype=dtype_var, like=backend_var)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0.0, 2 * D.pi), dt=0.01, rtol=D.epsilon(dtype_var)**0.75,
                     atol=D.epsilon(dtype_var)**0.75)
    a.set_kick_vars(D.ar_numpy.array([0,1], dtype=D.autoray.to_backend_dtype('bool', like=backend_var), like=backend_var))
    if integrator is None:
        integrator = a.method
    else:
        a.method = integrator
    dt = D.tol_epsilon(dtype_var)**(1.0/(2+a.integrator.order))/(2*D.pi)
    a.dt = dt

    return de_mat, rhs, analytic_soln, y_init, dt, a
