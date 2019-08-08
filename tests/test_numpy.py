from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['DES_BACKEND']          = 'numpy'



import desolver as de
import desolver.backend as D




D.set_float_fmt('float64')

de_mat = D.array([[0.0, 1.0],[-1.0, 0.0]])

@de.rhs_prettifier("""[vx, -x+t]""")
def rhs(t, state, **kwargs):    
    return de_mat @ state + D.array([0.0, t])

def analytic_soln(t, initial_conditions):
    c1 = initial_conditions[0]
    c2 = initial_conditions[1] - 1
    
    return D.array([
        c2 * D.sin(t) + c1 * D.cos(t) + t,
        c2 * D.cos(t) - c1 * D.sin(t) + 1
    ])

def kbinterrupt_cb(ode_sys):
    if ode_sys[-1].t > D.pi:
        raise KeyboardInterrupt("Test Interruption and Catching")

y_init = D.array([1., 0.])

a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi - 0.01), dt=0.01, rtol=5e-6, atol=5e-6)

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
    for i in sorted(set(de.available_methods.values()), key=lambda x:x.__name__):
        try:
            a.set_method(i)
            print("Testing {}".format(a.integrator))
            try:
                a.integrate(callback=kbinterrupt_cb, eta=True)
            except KeyboardInterrupt as e:
                print("")
                print(e)
                print(a.integration_status())
                print("")
            a.integrate(eta=True)

            max_diff = D.max(D.abs(analytic_soln(a.t[-1], a.y[0])-a.y[-1]))
            if a.method.__adaptive__ and max_diff >= a.atol * 10 + D.epsilon():
                print("{} Failed with max_diff from analytical solution = {}".format(a.integrator, max_diff))
                raise RuntimeError("Failed to meet tolerances for adaptive integrator {}".format(str(i)))
            else:
                print("{} Succeeded with max_diff from analytical solution = {}".format(a.integrator, max_diff))
                
                
                
            a.reset()
        except:
            raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
    print("")

print("{} backend test passed successfully!".format(os.environ['DES_BACKEND']))