import pytest
import desolver as de
import desolver.backend as D
import numpy as np
from .common import ffmt_param, integrator_param, richardson_param, device_param

@ffmt_param
@integrator_param
@richardson_param
@device_param
def test_float_formats_typical_shape(ffmt, integrator, use_richardson_extrapolation, device):
    D.set_float_fmt(ffmt)

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(False) # Enable if a test fails
        
        device = torch.device(device)

    print("Testing {} float format".format(D.float_fmt()))

    from .common import set_up_basic_system

    de_mat, rhs, analytic_soln, y_init, dt, _ = set_up_basic_system(integrator, hook_jacobian=True)

    y_init = D.array([1., 0.])
    
    if D.backend() == 'torch':
        y_init = y_init.to(device)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, D.pi / 4), dt=D.pi/64, rtol=D.epsilon() ** 0.5,
                     atol=D.epsilon() ** 0.5)

    method = integrator
    method_tolerance = a.atol * 10 + D.epsilon()
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        method_tolerance = method_tolerance * 5
    
    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        a.set_method(method)
        print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))

        a.integrate(eta=True)

        print("Average step-size:", D.mean(D.abs(D.array(a.t[1:]) - D.array(a.t[:-1]))))
        max_diff = D.max(D.abs(analytic_soln(a.t[-1], y_init) - a.y[-1]))
        if a.integrator.adaptive:
            assert max_diff <= method_tolerance, "{} Failed with max_diff from analytical solution = {}".format(a.integrator, max_diff)
        if a.integrator.__implicit__:
            assert rhs.analytic_jacobian_called and a.njev > 0, "Analytic jacobian was called as part of integration"
        a.reset()
    print("")

    print("{} backend test passed successfully!".format(D.backend()))
    

@ffmt_param
@integrator_param
@richardson_param
@device_param
def test_float_formats_atypical_shape(ffmt, integrator, use_richardson_extrapolation, device):
    D.set_float_fmt(ffmt)

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(False) # Enable if a test fails
        
        device = torch.device(device)

    print("Testing {} float format".format(D.float_fmt()))

    from .common import set_up_basic_system

    de_mat, _, analytic_soln, y_init, dt, _ = set_up_basic_system(integrator)
    
    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, **kwargs):
        nonlocal de_mat
        extra = D.array([0.0, t])
        if D.backend() == 'torch':
            de_mat = de_mat.to(state.device)
            extra  = extra.to(state.device)
        return D.sum(de_mat[:, :, None, None, None] * state, axis=1) + extra[:, None, None, None]

    y_init = D.array([[[[1., 0.]]*1]*1]*3).T
    
    if D.backend() == 'torch':
        y_init = y_init.contiguous().to(device)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, D.pi / 4), dt=D.pi/64, rtol=D.epsilon()**0.5,
                     atol=D.epsilon()**0.5)

    method = integrator
    method_tolerance = a.atol * 10 + D.epsilon()
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        method_tolerance = method_tolerance * 5
    
    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        a.set_method(method)
        print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))

        a.integrate(eta=True)

        max_diff = D.max(D.abs(analytic_soln(a.t[-1], y_init) - a.y[-1]))
        if a.integrator.adaptive:
            assert max_diff <= method_tolerance, "{} Failed with max_diff from analytical solution = {}".format(a.integrator, max_diff)
        a.reset()
    print("")

    print("{} backend test passed successfully!".format(D.backend()))