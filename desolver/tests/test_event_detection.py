import desolver as de
import desolver.backend as D
import numpy as np
import pytest

integrator_set = set(de.available_methods(False).values())
integrator_set = sorted(integrator_set, key=lambda x: x.__name__)
# integrator_set = [
#     intg if intg.__order__ > 1 else pytest.param(intg, marks=pytest.mark.xfail(reason=f"{intg.__name__} is too low order")) for intg in integrator_set
# ]


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator', integrator_set)
@pytest.mark.parametrize('use_richardson_extrapolation', [False, True])
def test_event_detection_multiple(ffmt, integrator, use_richardson_extrapolation):
    if ffmt == 'float16':
        return
    D.set_float_fmt(ffmt)

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(True)

    print("Testing event detection for float format {}".format(D.float_fmt()))

    from .common import set_up_basic_system

    de_mat, rhs, analytic_soln, y_init, dt, _ = set_up_basic_system(integrator)

    def time_event(t, y, **kwargs):
        return t - D.pi / 8

    def second_time_event(t, y, **kwargs):
        return t - D.pi / 16

    def first_y_event(t, y, **kwargs):
        return y[0] - analytic_soln(D.pi / 32, y_init)[0]

    time_event.is_terminal = True
    time_event.direction = 0
    second_time_event.is_terminal = False
    second_time_event.direction = 0
    first_y_event.is_terminal = False
    first_y_event.direction = 0

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi / 4), dt=0.01, rtol=D.epsilon() ** 0.5,
                     atol=D.epsilon() ** 0.5)
    a.set_kick_vars(D.array([0,1],dtype=D.bool))

    method = integrator
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        
    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        a.set_method(method)
        print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))
        assert (a.integration_status() == "Integration has not been run.")

        a.integrate(eta=False, events=[time_event, second_time_event, first_y_event])

        print(a.events, D.pi / 32, analytic_soln(D.pi/32, y_init), analytic_soln(D.pi/32, y_init)[0] - a.events[0].y)
        try:
            assert (D.abs(a.events[0].y[0] - analytic_soln(D.pi/16, y_init)[0]) <= 10 * (D.epsilon()**0.1))
            assert (D.abs(a.events[1].t - D.pi / 16) <= 10 * D.epsilon())
            assert (D.abs(a.events[2].t - D.pi / 8) <= 10 * D.epsilon())
            assert (D.abs(a.t[-1] - D.pi / 8) <= 10 * D.epsilon())
            assert (len(a.events) == 3)
        except:
            print("Event detection with integrator {} failed with t[-1] = {}".format(a.integrator, a.t[-1]))
            raise RuntimeError("Failed to detect event for integrator {}".format(str(method)))
        else:
            print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
        a.reset()
    print("")

    print("{} backend test passed successfully!".format(D.backend()))


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator', integrator_set)
@pytest.mark.parametrize('use_richardson_extrapolation', [False, True])
def test_event_detection_single(ffmt, integrator, use_richardson_extrapolation):
    if ffmt == 'float16':
        return

    D.set_float_fmt(ffmt)

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(True)

    print("Testing event detection for float format {}".format(D.float_fmt()))
    from .common import set_up_basic_system

    de_mat, rhs, analytic_soln, y_init, dt, _ = set_up_basic_system(integrator)

    def time_event(t, y, **kwargs):
        return t - D.pi / 8

    time_event.is_terminal = True
    time_event.direction = 0

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi / 4), dt=0.01, rtol=D.epsilon() ** 0.5,
                     atol=D.epsilon() ** 0.5)

    method = integrator
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        
    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        a.set_method(method)
        print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))
        assert (a.integration_status() == "Integration has not been run.")

        a.integrate(eta=False, events=time_event)

        assert (a.integration_status() == "Integration terminated upon finding a triggered event.")

        try:
            assert (D.abs(a.t[-1] - D.pi / 8) <= 10 * D.epsilon())
            assert (len(a.events) == 1)
        except:
            print("Event detection with integrator {} failed with t[-1] = {}".format(a.integrator, a.t[-1]))
            raise RuntimeError("Failed to detect event for integrator {}".format(str(method)))
        else:
            print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
        a.reset()
    print("")

    print("{} backend test passed successfully!".format(D.backend()))
