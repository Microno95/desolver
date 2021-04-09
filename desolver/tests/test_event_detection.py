import desolver as de
import desolver.backend as D
import numpy as np
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

@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
@pytest.mark.parametrize('use_richardson_extrapolation', [False, True])
@pytest.mark.parametrize('device', devices_set)
def test_event_detection_multiple(ffmt, integrator, use_richardson_extrapolation, device):
    if integrator.__implicit__ and use_richardson_extrapolation:
        pytest.skip("Implicit methods are unstable with richardson extrapolation")
    if ffmt == 'float16':
        pytest.skip("Event detection is unstable with {}".format(ffmt))
    D.set_float_fmt(ffmt)

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(False) # Enable if a test fails
        
        device = torch.device(device)

    print("Testing event detection for float format {}".format(D.float_fmt()))

    from .common import set_up_basic_system

    de_mat, rhs, analytic_soln, y_init, dt, _ = set_up_basic_system(integrator, hook_jacobian=True)

    if D.backend() == 'torch':
        y_init = y_init.to(device)

    def time_event(t, y, **kwargs):
        out = D.array(t - D.pi / 2)
        if D.backend() == 'torch':
            out = out.to(device)
        return out

    def second_time_event(t, y, **kwargs):
        out = D.array(t - D.pi / 4)
        if D.backend() == 'torch':
            out = out.to(device)
        return out

    def first_y_event(t, y, **kwargs):
        return y[0] - analytic_soln(D.pi / 8, y_init)[0]

    time_event.is_terminal = True
    time_event.direction = 0
    second_time_event.is_terminal = False
    second_time_event.direction = 0
    first_y_event.is_terminal = False
    first_y_event.direction = 0

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi), dt=D.pi / 128, rtol=min(1e-3, D.epsilon()**0.5),
                     atol=min(1e-3, D.epsilon()**0.5))
    a.set_kick_vars(D.array([0,1],dtype=D.bool))

    method = integrator
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        
    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        a.set_method(method)
        print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))
        assert (a.integration_status == "Integration has not been run.")

        a.integrate(eta=True, events=[time_event, second_time_event, first_y_event])

        print(a.events)
        print(D.pi / 8, analytic_soln(D.pi/8, y_init), analytic_soln(D.pi/8, y_init)[0] - a.events[0].y[0])
        assert (len(a.events) == 3)
        assert (D.abs(a.events[0].y[0] - D.array(analytic_soln(D.pi/8, y_init)[0])) <= 10 * (D.epsilon()**0.1))
        assert (a.events[0].event is first_y_event)
        assert (D.abs(a.events[1].t - D.array(D.pi / 4)) <= 10 * (D.epsilon()**0.25))
        assert (a.events[1].event is second_time_event)
        assert (D.abs(a.events[2].t - D.array(D.pi / 2)) <= 10 * (D.epsilon()**0.25))
        assert (a.events[2].event is time_event)
        assert (D.abs(a.t[-1] - D.array(D.pi / 2)) <= 10 * (D.epsilon()**0.25))
        print("Event detection with integrator {} succeeded with t[-1] = {}, expected = {}, diff = {}".format(a.integrator, a.t[-1], D.pi / 2, a.t[-1] - D.pi / 2))
        a.reset()
    print("")

    print("{} backend test passed successfully!".format(D.backend()))


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
@pytest.mark.parametrize('use_richardson_extrapolation', [False, True])
@pytest.mark.parametrize('device', devices_set)
def test_event_detection_single(ffmt, integrator, use_richardson_extrapolation, device):
    if integrator.__implicit__ and use_richardson_extrapolation:
        pytest.skip("Implicit methods are unstable with richardson extrapolation")
    if ffmt == 'float16':
        pytest.skip("Event detection is unstable with {}".format(ffmt))

    D.set_float_fmt(ffmt)

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(False) # Enable if a test fails
        
        device = torch.device(device)

    print("Testing event detection for float format {}".format(D.float_fmt()))
    from .common import set_up_basic_system

    de_mat, rhs, analytic_soln, y_init, dt, _ = set_up_basic_system(integrator, hook_jacobian=True)

    if D.backend() == 'torch':
        y_init = y_init.to(device)

    def time_event(t, y, **kwargs):
        out = D.array(t - D.pi / 8)
        if D.backend() == 'torch':
            out = out.to(device)
        return out

    time_event.is_terminal = True
    time_event.direction = 0

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi / 4), dt=D.pi / 64, rtol=min(1e-3, D.epsilon()**0.5),
                     atol=min(1e-3, D.epsilon()**0.5))

    method = integrator
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        
    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        a.set_method(method)
        print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))
        assert (a.integration_status == "Integration has not been run.")

        a.integrate(eta=True, events=time_event)

        assert (a.integration_status == "Integration terminated upon finding a triggered event.")

        assert (D.abs(a.t[-1] - D.pi / 8) <= 10 * D.epsilon())
        assert (len(a.events) == 1)
        assert (a.events[0].event == time_event)
        print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
        a.reset()
    print("")

    print("{} backend test passed successfully!".format(D.backend()))


# @pytest.mark.parametrize('ffmt', D.available_float_fmt())
# @pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
# @pytest.mark.parametrize('use_richardson_extrapolation', [False, True])
# @pytest.mark.parametrize('device', devices_set)
# def test_event_detection_multiple_roots(ffmt, integrator, use_richardson_extrapolation, device):
#     if integrator.__implicit__ and use_richardson_extrapolation:
#         pytest.skip("Implicit methods are unstable with richardson extrapolation")
#     if ffmt == 'float16':
#         pytest.skip("Event detection is unstable with {}".format(ffmt))

#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(False) # Enable if a test fails
        
#         device = torch.device(device)

#     print("Testing event detection for float format {}".format(D.float_fmt()))
#     from .common import set_up_basic_system

#     de_mat, rhs, analytic_soln, y_init, dt, _ = set_up_basic_system(integrator, hook_jacobian=True)

#     if D.backend() == 'torch':
#         y_init = y_init.to(device)

#     def time_event(t, y, **kwargs):
#         out = D.array((t - D.pi / 8)*(t - D.pi / 16)*(t - D.pi / 32))
#         if D.backend() == 'torch':
#             out = out.to(device)
#         return out

#     time_event.is_terminal = False
#     time_event.direction = 0

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi / 4), dt=D.pi / 64, rtol=min(1e-3, D.epsilon()**0.5),
#                      atol=min(1e-3, D.epsilon()**0.5))

#     method = integrator
#     if use_richardson_extrapolation:
#         method = de.integrators.generate_richardson_integrator(method)
        
#     with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
#         a.set_method(method)
#         print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))
#         assert (a.integration_status == "Integration has not been run.")

#         a.integrate(eta=True, events=time_event)

#         print(a.events)
#         assert (D.abs(a.t[-1] - D.pi / 8) <= 10 * D.epsilon())
#         assert (len(a.events) == 3)
#         print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
#         a.reset()
#     print("")

#     print("{} backend test passed successfully!".format(D.backend()))
