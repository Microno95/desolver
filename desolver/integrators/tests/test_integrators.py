from desolver.tests.common import ffmt_param, device_param, integrator_set
from desolver.integrators.integrator_types import RungeKuttaIntegrator
import desolver.backend as D
import pytest

@ffmt_param
@device_param
@pytest.mark.parametrize("integrator", integrator_set)
def test_instantiation(ffmt, device, integrator):
    D.set_float_fmt(ffmt)

    integrator_instance = integrator(sys_dim=(1,), dtype=D.zeros((1,)).dtype, rtol=None, atol=None, device=device)

    assert integrator_instance.rtol == 32 * D.epsilon(), "Relative tolerance is incorrect!"
    assert integrator_instance.atol == 32 * D.epsilon(), "Absolute tolerance is incorrect!"
    assert integrator_instance.numel == 1, "Integrator dimensions should be prod(sys_dim) = prod((1,)) = 1!"
    assert D.all(integrator_instance.dState == 0.0), "State delta must be zero upon init!"
    assert D.all(integrator_instance.dTime == 0.0), "Time delta must be zero upon init!"
    assert integrator_instance.device == device, "Integrator device is incorrect!"
    if hasattr(integrator_instance, "final_state"):
        assert integrator_instance.adaptive == (integrator_instance.final_state.shape[0] == 2)
    else:
        assert integrator_instance.adaptive == False
    if issubclass(integrator, RungeKuttaIntegrator):
        assert integrator_instance.explicit == all([D.all(integrator_instance.tableau[col, col+1:] == 0.0) for col in range(integrator_instance.tableau.shape[0])])
        assert integrator_instance.fsal == D.all(integrator_instance.tableau[-1, 1:] == integrator_instance.final_state[0, 1:])


