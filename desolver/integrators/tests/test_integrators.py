import desolver.backend as D
from desolver.integrators import IntegratorTemplate, RungeKuttaIntegrator
import pytest


def test_instantiation(dtype_var, backend_var, device_var, integrators: IntegratorTemplate|RungeKuttaIntegrator):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)

    integrator_instance = integrators(sys_dim=(1,), dtype=dtype_var, rtol=None, atol=None, device=device_var)

    assert integrator_instance.rtol == 32 * D.epsilon(dtype_var), "Relative tolerance is incorrect!"
    assert integrator_instance.atol == 32 * D.epsilon(dtype_var), "Absolute tolerance is incorrect!"
    assert integrator_instance.numel == 1, "Integrator dimensions should be prod(sys_dim) = prod((1,)) = 1!"
    assert D.ar_numpy.all(integrator_instance.dState == 0.0), "State delta must be zero upon init!"
    assert D.ar_numpy.all(integrator_instance.dTime == 0.0), "Time delta must be zero upon init!"
    assert integrator_instance.device == device_var, "Integrator device is incorrect!"
    
    if issubclass(integrators, RungeKuttaIntegrator):
        assert integrator_instance.is_adaptive == (integrator_instance.tableau_final.shape[0] == 2)
        assert integrator_instance.is_explicit == all([D.ar_numpy.all(integrator_instance.tableau_intermediate[col, col+1:] == 0.0) for col in range(integrator_instance.tableau_intermediate.shape[0])])
        assert integrator_instance.is_fsal == D.ar_numpy.all(integrator_instance.tableau_intermediate[-1, 1:] == integrator_instance.tableau_final[0, 1:])
