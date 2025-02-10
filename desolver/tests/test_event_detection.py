import desolver as de
import desolver.backend as D
import numpy as np
import pytest
from copy import deepcopy
from desolver.tests import common


@common.basic_integrator_param
@common.dense_output_param
def test_event_detection_single(dtype_var, backend_var, integrator, dense_output):
    if "float16" in dtype_var:
        pytest.skip("Event detection with 'float16' types are unreliable due to imprecision")
    
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)

    de_mat, rhs, analytic_soln, y_init, _, _ = common.set_up_basic_system(dtype_var, backend_var, integrator, hook_jacobian=True)

    def time_event(t, y, **kwargs):
        out = D.ar_numpy.asarray(t - D.pi / 8, dtype=dtype_var, like=y_init)
        if backend_var == 'torch':
            out = out.to(y_init)
        return out

    time_event.is_terminal = True
    time_event.direction = 0

    a = de.OdeSystem(rhs, y0=y_init, dense_output=dense_output, t=(0, D.pi / 4), dt=D.pi/512, rtol=D.epsilon(dtype_var) ** 0.5,
                     atol=D.epsilon(dtype_var) ** 0.5)
    
    a.set_kick_vars([0, 1])

    method = integrator

    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        a.set_method(method)

        a.integrate(eta=False, events=time_event)

        assert (a.integration_status == "Integration terminated upon finding a triggered event.")

        print(a)
        print(a.events)
        assert (D.ar_numpy.abs(a.t[-1] - D.pi / 8) <= D.epsilon(dtype_var) ** 0.5)
        assert (len(a.events) == 1)
        assert (a.events[0].event == time_event)
        print("Event detection with integrator {} succeeded with t[-1] = {}, diff = {}".format(a.integrator, a.t[-1],
                                                                                               a.t[-1] - D.pi / 8))


@common.basic_integrator_param
@common.dense_output_param
def test_event_detection_multiple(dtype_var, backend_var, integrator, dense_output):
    if "float16" in dtype_var:
        pytest.skip("Event detection with 'float16' types are unreliable due to imprecision")
    
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        import torch
        torch.set_printoptions(precision=17)
        torch.autograd.set_detect_anomaly(True)
        
    array_con_args = dict(dtype=dtype_var, like=backend_var)

    de_mat, rhs, analytic_soln, y_init, _, _ = common.set_up_basic_system(dtype_var, backend_var, integrator, hook_jacobian=True)


    def time_event(t, y, **kwargs):
        out = D.ar_numpy.asarray(t - D.pi / 2, dtype=dtype_var, like=y_init)
        if backend_var == 'torch':
            out = out.to(y_init)
        return out

    def second_time_event(t, y, **kwargs):
        out = D.ar_numpy.asarray(t - D.pi / 4, dtype=dtype_var, like=y_init)
        if backend_var == 'torch':
            out = out.to(y_init)
        return out

    def first_y_event(t, y, **kwargs):
        return y[0] - analytic_soln(D.pi / 8, y_init)[0]

    time_event.is_terminal = True
    time_event.direction = 0
    second_time_event.is_terminal = False
    second_time_event.direction = 0
    first_y_event.is_terminal = False
    first_y_event.direction = 0

    a = de.OdeSystem(rhs, y0=y_init, dense_output=dense_output, t=(0, D.pi), dt=D.pi/512, rtol=D.epsilon(dtype_var) ** 0.5,
                     atol=D.epsilon(dtype_var) ** 0.5)
    
    a.set_kick_vars([0, 1])

    method = integrator

    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        a.set_method(method)

        def dt_max_cb(ode_sys):
            ode_sys.dt = min(ode_sys.dt, D.pi / 8)
        a.integrate(eta=False, events=[time_event, second_time_event, first_y_event], callback=[dt_max_cb])

        assert (a.integration_status == "Integration terminated upon finding a triggered event.")

        print(a.events)
        print(D.pi / 8, analytic_soln(D.pi / 8, y_init), analytic_soln(D.pi / 8, y_init)[0] - a.events[0].y[0])
        assert (len(a.events) == 3)
        assert (D.ar_numpy.abs(D.ar_numpy.to_numpy(a.events[0].y[0]) - D.ar_numpy.to_numpy(analytic_soln(D.pi / 8, y_init)[0])) <= (
                    32 * D.epsilon(dtype_var) ** 0.5))
        assert (a.events[0].event is first_y_event)
        assert (D.ar_numpy.abs(a.events[1].t - D.ar_numpy.asarray(D.pi / 4, **array_con_args)) <= (D.epsilon(dtype_var) ** 0.5))
        assert (a.events[1].event is second_time_event)
        assert (D.ar_numpy.abs(a.events[2].t - D.ar_numpy.asarray(D.pi / 2, **array_con_args)) <= (D.epsilon(dtype_var) ** 0.5))
        assert (a.events[2].event is time_event)
        assert (D.ar_numpy.abs(a.t[-1] - D.ar_numpy.asarray(D.pi / 2, **array_con_args)) <= (D.epsilon(dtype_var) ** 0.5))
        print("Event detection with integrator {} succeeded with t[-1] = {}, diff = {}".format(a.integrator, a.t[-1],
                                                                                               a.t[-1] - D.pi / 2))

# @ffmt_param
# @integrator_param
# @richardson_param
# @device_param
# @dt_param
# # @dense_output_param
# def test_event_detection_multiple(ffmt, integrator, use_richardson_extrapolation, device, dt, dense_output=False):
#     if use_richardson_extrapolation and integrator.is_implicit():
#         pytest.skip("Richardson Extrapolation is too slow with implicit methods")
#     if ffmt == 'gdual_vdouble':
#         pytest.skip("gdual_vdouble type does not support event detection")
#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(False)  # Enable if a test fails

#         device = torch.device(device)

#     print("Testing event detection for float format {}".format(D.float_fmt()))

#     from .common import set_up_basic_system

#     de_mat, rhs, analytic_soln, y_init, _, _ = set_up_basic_system(integrator, hook_jacobian=True)

#     if D.backend() == 'torch':
#         y_init = y_init.to(device)

#     def time_event(t, y, **kwargs):
#         out = D.ar_numpy.asarray(t - D.pi / 2)
#         if D.backend() == 'torch':
#             out = out.to(device)
#         return out

#     def second_time_event(t, y, **kwargs):
#         out = D.ar_numpy.asarray(t - D.pi / 4)
#         if D.backend() == 'torch':
#             out = out.to(device)
#         return out

#     def first_y_event(t, y, **kwargs):
#         return y[0] - analytic_soln(D.pi / 8, y_init)[0]

#     time_event.is_terminal = True
#     time_event.direction = 0
#     second_time_event.is_terminal = False
#     second_time_event.direction = 0
#     first_y_event.is_terminal = False
#     first_y_event.direction = 0

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=dense_output, t=(0, D.pi), dt=dt, rtol=D.epsilon(dtype_var) ** 0.5,
#                      atol=D.epsilon(dtype_var) ** 0.75)
#     a.set_kick_vars(D.ar_numpy.asarray([0, 1], dtype=D.bool))

#     method = integrator
#     if use_richardson_extrapolation:
#         method = de.integrators.generate_richardson_integrator(method)

#     with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
#         a.set_method(method)
#         print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))

#         def dt_max_cb(ode_sys):
#             ode_sys.dt = min(ode_sys.dt, D.pi / 8)
#         a.integrate(eta=True, events=[time_event, second_time_event, first_y_event], callback=[dt_max_cb])

#         assert (a.integration_status == "Integration terminated upon finding a triggered event.")

#         print(a.events)
#         print(D.pi / 8, analytic_soln(D.pi / 8, y_init), analytic_soln(D.pi / 8, y_init)[0] - a.events[0].y[0])
#         assert (len(a.events) == 3)
#         assert (D.ar_numpy.abs(D.to_float(a.events[0].y[0]) - D.to_float(analytic_soln(D.pi / 8, y_init)[0])) <= (
#                     32 * D.epsilon(dtype_var) ** 0.5))
#         assert (a.events[0].event is first_y_event)
#         assert (D.ar_numpy.abs(a.events[1].t - D.ar_numpy.asarray(D.pi / 4)) <= (D.epsilon(dtype_var) ** 0.5))
#         assert (a.events[1].event is second_time_event)
#         assert (D.ar_numpy.abs(a.events[2].t - D.ar_numpy.asarray(D.pi / 2)) <= (D.epsilon(dtype_var) ** 0.5))
#         assert (a.events[2].event is time_event)
#         assert (D.ar_numpy.abs(a.t[-1] - D.ar_numpy.asarray(D.pi / 2)) <= (D.epsilon(dtype_var) ** 0.5))
#         print("Event detection with integrator {} succeeded with t[-1] = {}, diff = {}".format(a.integrator, a.t[-1],
#                                                                                                a.t[-1] - D.pi / 2))


# @ffmt_param
# @integrator_param
# @richardson_param
# @device_param
# @dt_param
# # @dense_output_param
# def test_event_detection_close_roots(ffmt, integrator, use_richardson_extrapolation, device, dt, dense_output=False):
#     if use_richardson_extrapolation and integrator.is_implicit():
#         pytest.skip("Richardson Extrapolation is too slow with implicit methods")
#     if ffmt == 'gdual_vdouble':
#         pytest.skip("gdual_vdouble type does not support event detection")
#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(False)  # Enable if a test fails

#         device = torch.device(device)

#     print("Testing event detection for float format {}".format(D.float_fmt()))
#     from .common import set_up_basic_system

#     de_mat, rhs, analytic_soln, y_init, _, _ = set_up_basic_system(integrator, hook_jacobian=True)

#     if D.backend() == 'torch':
#         y_init = y_init.to(device)

#     def first_y_event(t, y, **kwargs):
#         return y[0] - analytic_soln(D.pi / 8, y_init)[0]

#     def second_y_event(t, y, **kwargs):
#         return y[0] - analytic_soln(D.pi / 7.5, y_init)[0]

#     def third_y_event(t, y, **kwargs):
#         return y[0] - analytic_soln(D.pi / 7, y_init)[0]

#     third_y_event.is_terminal = True

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=dense_output, t=(0, D.pi / 6), dt=dt, rtol=D.epsilon(dtype_var) ** 0.5,
#                      atol=D.epsilon(dtype_var) ** 0.75)
#     a.set_kick_vars(D.ar_numpy.asarray([0, 1], dtype=bool))

#     method = integrator
#     if use_richardson_extrapolation:
#         method = de.integrators.generate_richardson_integrator(method)

#     with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
#         a.set_method(method)
#         print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))

#         a.integrate(eta=True, events=[first_y_event, second_y_event, third_y_event])

#         assert (a.integration_status == "Integration terminated upon finding a triggered event.")

#         print(a)
#         print(a.events)
#         print(D.pi / 7.0)
#         print(D.pi / 7.5)
#         print(D.pi / 8.0)
#         assert (len(a.events) == 3)
#         assert (D.ar_numpy.abs(
#             D.to_float(a.events[2].y[0]) - D.to_float(analytic_soln(D.pi / 7, y_init)[0])) <= D.epsilon(dtype_var) ** 0.5)
#         assert (D.ar_numpy.abs(
#             D.to_float(a.events[1].y[0]) - D.to_float(analytic_soln(D.pi / 7.5, y_init)[0])) <= D.epsilon(dtype_var) ** 0.5)
#         assert (D.ar_numpy.abs(
#             D.to_float(a.events[0].y[0]) - D.to_float(analytic_soln(D.pi / 8, y_init)[0])) <= D.epsilon(dtype_var) ** 0.5)
#         assert (a.events[2].event == third_y_event)
#         assert (a.events[1].event == second_y_event)
#         assert (a.events[0].event == first_y_event)
#         print("Event detection with integrator {} succeeded with t[-1] = {}, diff = {}".format(a.integrator, a.t[-1],
#                                                                                                a.t[-1] - D.pi / 7))


# @pytest.mark.skip("Too slow, needs refactoring")
# @ffmt_param
# @integrator_param
# @richardson_param
# @device_param
# @dt_param
# # @dense_output_param
# def test_event_detection_numerous_events(ffmt, integrator, use_richardson_extrapolation, device, dt,
#                                          dense_output=False):
#     if use_richardson_extrapolation and integrator.is_implicit():
#         pytest.skip("Richardson Extrapolation is too slow with implicit methods")
#     if ffmt == 'gdual_vdouble':
#         pytest.skip("gdual_vdouble type does not support event detection")
#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(False)  # Enable if a test fails

#         device = torch.device(device)

#     print("Testing event detection for float format {}".format(D.float_fmt()))
#     from .common import set_up_basic_system

#     de_mat, rhs, analytic_soln, y_init, _, _ = set_up_basic_system(integrator, hook_jacobian=True)

#     if D.backend() == 'torch':
#         y_init = y_init.to(device)

#     event_times = D.linspace(0, D.pi / 4, 32)

#     class ev_proto:
#         def __init__(self, ev_time, component):
#             self.ev_time = ev_time
#             self.component = component

#         def __call__(self, t, y, **csts):
#             return y[self.component] - analytic_soln(self.ev_time, y_init)[self.component]

#         def __repr__(self):
#             return "<ev_proto({}, {})>".format(self.ev_time, self.component)

#     events = [
#         ev_proto(ev_t, 0) for ev_t in event_times
#     ]

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=dense_output, t=(0, D.pi / 4), dt=dt, rtol=D.epsilon(dtype_var) ** 0.5,
#                      atol=D.epsilon(dtype_var) ** 0.75)

#     method = integrator
#     if use_richardson_extrapolation:
#         method = de.integrators.generate_richardson_integrator(method)

#     with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
#         a.set_method(method)
#         print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))

#         a.integrate(eta=True, events=events)

#         print(a)
#         print(a.events)
#         print(len(events) - len(a.events))
#         assert (len(events) - 3 <= len(a.events) <= len(events))
#         for ev_detected in a.events:
#             assert (D.max(
#                 D.ar_numpy.abs(D.to_float(ev_detected.event(ev_detected.t, ev_detected.y, **a.constants)))) <= 4 * D.epsilon(dtype_var))

# @pytest.mark.parametrize('ffmt', D.available_float_fmt())
# @pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
# @pytest.mark.parametrize('use_richardson_extrapolation', [False, True])
# @pytest.mark.parametrize('device', device_var_set)
# def test_event_detection_multiple_roots(ffmt, integrator, use_richardson_extrapolation, device):
#     if integrator.is_implicit() and use_richardson_extrapolation:
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

#     de_mat, rhs, analytic_soln, y_init, _, _ = set_up_basic_system(integrator, hook_jacobian=True)

#     if D.backend() == 'torch':
#         y_init = y_init.to(device)

#     def time_event(t, y, **kwargs):
#         out = D.ar_numpy.asarray((t - D.pi / 8)*(t - D.pi / 16)*(t - D.pi / 32))
#         if D.backend() == 'torch':
#             out = out.to(device)
#         return out

#     time_event.is_terminal = False
#     time_event.direction = 0

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi / 4), dt=D.pi / 64, rtol=min(1e-3, D.epsilon(dtype_var)**0.5),
#                      atol=min(1e-3, D.epsilon(dtype_var)**0.5))

#     method = integrator
#     if use_richardson_extrapolation:
#         method = de.integrators.generate_richardson_integrator(method)

#     with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
#         a.set_method(method)
#         print("Testing {} with dt = {:.4e}".format(a.integrator, a.dt))
#         assert (a.integration_status == "Integration has not been run.")

#         a.integrate(eta=True, events=time_event)

#         print(a.events)
#         assert (D.ar_numpy.abs(a.t[-1] - D.pi / 8) <= 10 * D.epsilon(dtype_var))
#         assert (len(a.events) == 3)
#         print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
#         a.reset()
#     print("")

#     print("{} backend test passed successfully!".format(D.backend()))
