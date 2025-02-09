import desolver as de
import desolver.backend as D
import numpy as np
import pytest
# from .common import ffmt_param


# @ffmt_param
# def test_getter_setters(ffmt):
#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(True)

#     print("Testing {} float format".format(D.float_fmt()))

#     de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

#     @de.rhs_prettifier("""[vx, -x+t]""")
#     def rhs(t, state, **kwargs):
#         return de_mat @ state + D.array([0.0, t])

#     y_init = D.array([1., 0.])

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon() ** 0.5,
#                      atol=D.epsilon() ** 0.5)

#     assert (a.t0 == 0)
#     assert (a.tf == 2 * D.pi)
#     assert (a.dt == 0.01)
#     assert (a.get_current_time() == a.t0)
#     assert (a.rtol == D.epsilon() ** 0.5)
#     assert (a.atol == D.epsilon() ** 0.5)
#     assert (D.norm(a.y[0] - y_init) <= 2 * D.epsilon())
#     assert (D.norm(a.y[-1] - y_init) <= 2 * D.epsilon())

#     a.set_kick_vars([True, False])

#     assert (a.staggered_mask == [True, False])
#     pval = 3 * D.pi

#     a.tf = pval

#     assert (a.tf == pval)
#     pval = -1.0

#     a.t0 = pval

#     assert (a.t0 == pval)
#     assert (a.dt == 0.01)

#     a.rtol = 1e-3

#     assert (a.rtol == 1e-3)

#     a.atol = 1e-3

#     assert (a.atol == 1e-3)

#     for method in de.available_methods():
#         a.set_method(method)
#         assert (isinstance(a.integrator, de.available_methods(False)[method]))

#     for method in de.available_methods():
#         a.method = method
#         assert (isinstance(a.integrator, de.available_methods(False)[method]))

#     a.constants['k'] = 5.0

#     assert (a.constants['k'] == 5.0)

#     a.constants.pop('k')

#     assert ('k' not in a.constants.keys())

#     new_constants = dict(k=10.0)

#     a.constants = new_constants

#     assert (a.constants['k'] == 10.0)

#     del a.constants

#     assert (not bool(a.constants))


# @ffmt_param
# def test_integration_and_representation(ffmt):
#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(True)

#     print("Testing {} float format".format(D.float_fmt()))

#     from . import common

#     (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system()

#     assert (a.integration_status == "Integration has not been run.")

#     a.integrate()

#     assert (a.integration_status == "Integration completed successfully.")

#     print(str(a))
#     print(repr(a))
#     assert (D.max(D.abs(D.to_float(a.sol(a.t[0])) - D.to_float(y_init))) <= 8 * D.epsilon() ** 0.5)
#     assert (D.max(D.abs(D.to_float(a.sol(a.t[-1])) - D.to_float(analytic_soln(a.t[-1], y_init)))) <= 8 * D.epsilon() ** 0.5)
#     assert (D.max(D.abs(D.to_float(a.sol(a.t).T) - D.to_float(analytic_soln(a.t, y_init)))) <= 8 * D.epsilon() ** 0.5)

#     for i in a:
#         assert (D.max(D.abs(D.to_float(i.y) - D.to_float(analytic_soln(i.t, y_init)))) <= 8 * D.epsilon() ** 0.5)

#     assert (len(a.y) == len(a))
#     assert (len(a.t) == len(a))


# @ffmt_param
# def test_integration_and_nearest_float_no_dense_output(ffmt):
#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(True)

#     print("Testing {} float format".format(D.float_fmt()))

#     de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

#     @de.rhs_prettifier("""[vx, -x+t]""")
#     def rhs(t, state, k, **kwargs):
#         return de_mat @ state + D.array([0.0, t])

#     y_init = D.array([1., 0.])

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon() ** 0.5,
#                      atol=D.epsilon() ** 0.5, constants=dict(k=1.0))

#     assert (a.integration_status == "Integration has not been run.")

#     a.integrate()
    
#     assert (a.sol is None)

#     assert (a.integration_status == "Integration completed successfully.")

#     assert (D.abs(a.t[-2] - a[2 * D.pi].t) <= D.abs(a.dt))


# @ffmt_param
# def test_no_events(ffmt):
#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(True)

#     print("Testing {} float format".format(D.float_fmt()))

#     de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

#     @de.rhs_prettifier("""[vx, -x+t]""")
#     def rhs(t, state, k, **kwargs):
#         return de_mat @ state + D.array([0.0, t])

#     y_init = D.array([1., 0.])

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon() ** 0.5,
#                      atol=D.epsilon() ** 0.5, constants=dict(k=1.0))

#     a.integrate()

#     assert (len(a.events) == 0)


# @ffmt_param
# def test_wrong_t0(ffmt):
#     with pytest.raises(ValueError):
#         D.set_float_fmt(ffmt)

#         if D.backend() == 'torch':
#             import torch

#             torch.set_printoptions(precision=17)

#             torch.autograd.set_detect_anomaly(True)

#         print("Testing {} float format".format(D.float_fmt()))

#         de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

#         @de.rhs_prettifier("""[vx, -x+t]""")
#         def rhs(t, state, k, **kwargs):
#             return de_mat @ state + D.array([0.0, t])

#         y_init = D.array([1., 0.])

#         a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon() ** 0.5,
#                          atol=D.epsilon() ** 0.5, constants=dict(k=1.0))

#         a.t0 = 2 * D.pi


# @ffmt_param
# def test_wrong_tf(ffmt):
#     with pytest.raises(ValueError):
#         D.set_float_fmt(ffmt)

#         if D.backend() == 'torch':
#             import torch

#             torch.set_printoptions(precision=17)

#             torch.autograd.set_detect_anomaly(True)

#         print("Testing {} float format".format(D.float_fmt()))

#         from . import common

#         (de_mat, rhs, analytic_soln, y_init, dt, a) = common.set_up_basic_system()

#         a.tf = 0.0


# @ffmt_param
# def test_not_enough_time_values(ffmt):
#     with pytest.raises(ValueError):
#         D.set_float_fmt(ffmt)

#         if D.backend() == 'torch':
#             import torch

#             torch.set_printoptions(precision=17)

#             torch.autograd.set_detect_anomaly(True)

#         print("Testing {} float format".format(D.float_fmt()))

#         de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

#         @de.rhs_prettifier("""[vx, -x+t]""")
#         def rhs(t, state, k, **kwargs):
#             return de_mat @ state + D.array([0.0, t])

#         y_init = D.array([1., 0.])

#         a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0,), dt=0.01, rtol=D.epsilon() ** 0.5,
#                          atol=D.epsilon() ** 0.5, constants=dict(k=1.0))

#         a.tf = 0.0


# @ffmt_param
# def test_dt_dir_fix(ffmt):
#     D.set_float_fmt(ffmt)

#     if D.backend() == 'torch':
#         import torch

#         torch.set_printoptions(precision=17)

#         torch.autograd.set_detect_anomaly(True)

#     print("Testing {} float format".format(D.float_fmt()))

#     de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

#     @de.rhs_prettifier("""[vx, -x+t]""")
#     def rhs(t, state, k, **kwargs):
#         return de_mat @ state + D.array([0.0, t])

#     y_init = D.array([1., 0.])

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=-0.01, rtol=D.epsilon() ** 0.5,
#                      atol=D.epsilon() ** 0.5, constants=dict(k=1.0))


# @ffmt_param
# def test_non_callable_rhs(ffmt):
#     with pytest.raises(TypeError):
#         D.set_float_fmt(ffmt)

#         if D.backend() == 'torch':
#             import torch

#             torch.set_printoptions(precision=17)

#             torch.autograd.set_detect_anomaly(True)

#         print("Testing {} float format".format(D.float_fmt()))

#         from . import common

#         (de_mat, rhs, analytic_soln, y_init, dt, _) = common.set_up_basic_system()

#         a = de.OdeSystem(de_mat, y0=y_init, dense_output=False, t=(0, 2 * D.pi), dt=dt, rtol=D.epsilon() ** 0.5,
#                          atol=D.epsilon() ** 0.5, constants=dict(k=1.0))

#         a.tf = 0.0


# def test_callback_called():
#     de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

#     @de.rhs_prettifier("""[vx, -x+t]""")
#     def rhs(t, state, **kwargs):
#         return de_mat @ state + D.array([0.0, t])

#     y_init = D.array([1., 0.])
    
#     callback_called = False
    
#     def callback(ode_sys):
#         nonlocal callback_called
#         if not callback_called and ode_sys.t[-1] > D.pi:
#             callback_called = True

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon() ** 0.5,
#                      atol=D.epsilon() ** 0.5)

#     a.integrate(callback=callback)
    
#     assert(callback_called)
    

# def test_keyboard_interrupt_caught():
#     de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

#     @de.rhs_prettifier("""[vx, -x+t]""")
#     def rhs(t, state, **kwargs):
#         return de_mat @ state + D.array([0.0, t])

#     y_init = D.array([1., 0.])
    
#     def kb_callback(ode_sys):
#         if ode_sys.t[-1] > D.pi:
#             raise KeyboardInterrupt()

#     a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon() ** 0.5,
#                      atol=D.epsilon() ** 0.5)

#     with pytest.raises(KeyboardInterrupt):
#         a.integrate(callback=kb_callback)
    
#     assert(a.integration_status == "A KeyboardInterrupt exception was raised during integration.")
    
    

# def test_DiffRHS():
#     def rhs(t, state, k, **kwargs):
#         return None

#     wrapped_rhs_no_repr = de.DiffRHS(rhs)

#     assert (str(wrapped_rhs_no_repr) == str(rhs))
#     assert (wrapped_rhs_no_repr._repr_markdown_() == str(rhs))

#     wrapped_rhs_no_equ_repr = de.DiffRHS(rhs, equ_repr=None, md_repr="1")

#     assert (str(wrapped_rhs_no_equ_repr) == str(rhs))
#     assert (wrapped_rhs_no_equ_repr._repr_markdown_() == "1")

#     wrapped_rhs_no_md_repr = de.DiffRHS(rhs, equ_repr="1", md_repr=None)

#     assert (str(wrapped_rhs_no_md_repr) == "1")
#     assert (wrapped_rhs_no_md_repr._repr_markdown_() == "1")

#     wrapped_rhs_both_repr = de.DiffRHS(rhs, equ_repr="1", md_repr="2")

#     assert (str(wrapped_rhs_both_repr) == "1")
#     assert (wrapped_rhs_both_repr._repr_markdown_() == "2")
