import desolver as de
import desolver.backend as D
import pytest


integrator_set = set(de.available_methods(False).values())
integrator_set = sorted(integrator_set, key=lambda x: x.__name__)
explicit_integrator_set = [
    pytest.param(intg, marks=pytest.mark.explicit) for intg in integrator_set if not intg.__implicit__
]
implicit_integrator_set = [
    pytest.param(intg, marks=pytest.mark.implicit) for intg in integrator_set if intg.__implicit__
]


if D.backend() == 'torch':
    devices_set = [pytest.param('cpu', marks=pytest.mark.cpu)]
    import torch
    if torch.cuda.is_available():
        devices_set.insert(0, pytest.param('cuda', marks=pytest.mark.gpu))
else:
    devices_set = [None]
    
dt_set   = [D.pi / 307, D.pi / 512]
ffmt_set = D.available_float_fmt()

ffmt_param         = pytest.mark.parametrize('ffmt', ffmt_set)
integrator_param   = pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
richardson_param   = pytest.mark.parametrize('use_richardson_extrapolation', [True, False])
device_param       = pytest.mark.parametrize('device', devices_set)
dt_param           = pytest.mark.parametrize('dt', dt_set)
dense_output_param = pytest.mark.parametrize('dense_output', [True, False])

def set_up_basic_system(integrator=None, hook_jacobian=False):
    de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, **kwargs):
        nonlocal de_mat
        if D.backend() == 'torch':
            de_mat = de_mat.to(state.device)
        out = de_mat @ state
        out[1] += t
        return out
    
    if hook_jacobian:
        def rhs_jac(t, state, **kwargs):
            nonlocal de_mat
            rhs.analytic_jacobian_called = True
            if D.backend() == 'torch':
                de_mat = de_mat.to(state.device)
            return de_mat

        rhs.hook_jacobian_call(rhs_jac)

    def analytic_soln(t, initial_conditions):
        c1 = initial_conditions[0]
        c2 = initial_conditions[1] - 1.0

        return D.stack([
            c2 * D.sin(D.to_float(D.asarray(t))) + c1 * D.cos(D.to_float(D.asarray(t))) + D.asarray(t),
            c2 * D.cos(D.to_float(D.asarray(t))) - c1 * D.sin(D.to_float(D.asarray(t))) + 1
        ])

    y_init = D.array([1., 0.])

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon()**0.75,
                     atol=D.epsilon()**0.75)
    a.set_kick_vars(D.array([0,1],dtype=D.bool))
    if integrator is None:
        integrator = a.method
    dt = (D.epsilon() ** 0.5)**(1.0/(2+integrator.order))/(2*D.pi)
    a.dt = dt

    return de_mat, rhs, analytic_soln, y_init, dt, a