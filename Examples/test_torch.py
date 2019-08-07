from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['DES_BACKEND']          = 'torch'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '9'

import desolver as de
import desolver.backend as D

D.set_float_fmt('float64')

de_mat = D.array([[0.0, 1.0],[-1.0, 0.0]], device='cuda:0')

@de.rhs_prettifier("""[vx, x]""")
def rhs(t, state, **kwargs):    
    return de_mat @ state + D.array([0.0, t], device='cuda:0')

def analytic_soln(t, initial_conditions):
    c1 = initial_conditions[0]
    c2 = initial_conditions[1] - 1
    
    return D.array([
        c2 * D.sin(t) + c1 * D.cos(t) + t,
        c2 * D.cos(t) - c1 * D.sin(t) + 1
    ], device='cuda:0')

def kbinterrupt_cb(ode_sys):
    if ode_sys[-1].t > D.pi:
        raise KeyboardInterrupt("Test Interruption and Catching")

y_init = D.array([1., 0.], requires_grad=True, device='cuda:0')

a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi - 0.01), dt=0.1, rtol=1e-3, atol=1e-6)

prev_val = a.get_start_time()
a.set_start_time(prev_val + 0.0)

if a.get_start_time() - prev_val > D.epsilon():
    raise ValueError("Start time mismatch, expected {}, got {}".format(prev_val, a.get_start_time()))

prev_val = a.get_end_time()
a.set_end_time(prev_val + 0.01)

if a.get_end_time() - prev_val > 0.01 + D.epsilon():
    raise ValueError("End time mismatch, expected {}, got {}".format(prev_val + 0.01, a.get_end_time()))

prev_val = a.get_step_size()
a.set_step_size(a.get_step_size() / 10)

if D.abs(a.get_step_size() - prev_val / 10) > D.epsilon():
    raise ValueError("Step size mismatch, expected {}, got {}".format(prev_val / 10, a.get_step_size()))

a.set_method(de.ischemes.RK45CKSolver)
a.show_system()

print("")

with de.BlockTimer(section_label="Integrator Tests") as sttimer:
    for i in sorted(de.available_methods):
        if de.available_methods[i].__adaptive__:
            continue
        print("Testing {}".format(str(i)))
        try:
            a.set_method(i)
            try:
                a.integrate(callback=kbinterrupt_cb, eta=True)
            except KeyboardInterrupt as e:
                print("")
                print(e)
                print(a.integration_status())
                print("")
            a.integrate(eta=True)

            max_diff = D.max(D.abs(analytic_soln(a.t[-1], a.y[0])-a.y[-1]))
            if a.method.__adaptive__:
                assert(max_diff < a.atol * 10)
            print("{} Succeeded with max_diff from analytical solution = {}".format(str(i), max_diff))
            print()
            with de.BlockTimer(section_label="Jacobian Computation Test"):
                print("Jacobian of the final state wrt the initial state:\n   ", D.jacobian(a.y[-1], a.y[0]))
                print("Jacobian should be:\n   ", D.eye(2))
            a.reset()
        except:
            raise RuntimeError("Test failed for integration method: {}".format(i))
    print("")

print("{} backend test passed successfully!".format(os.environ['DES_BACKEND']))