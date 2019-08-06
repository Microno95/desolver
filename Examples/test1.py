from __future__ import absolute_import, division, print_function, unicode_literals

import desolver as de
import desolver.backend as D

D.set_float_fmt('float64')


@de.rhs_prettifier("""[vx, x]""")
def rhs(t, state, **kwargs):
    x,vx = state

    dx  = vx
    dvx = -x

    return D.array([dx, dvx])

def analytic_soln(t, initial_conditions):
    c1 = initial_conditions[0]
    c2 = initial_conditions[1]
    
    return D.array([
        c2 * D.sin(t) + c1 * D.cos(t),
        c2 * D.cos(t) - c1 * D.sin(t)
    ])

def kbinterrupt_cb(ode_sys):
    if ode_sys.t[-1] > D.pi:
        raise KeyboardInterrupt("Test Interruption and Catching")

if D.backend() == 'torch':
    y_init = D.array([1., 0.], requires_grad=True)
else:
    y_init = D.array([1., 0.])

a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=1e-6, atol=1e-9)

a.set_end_time(a.get_end_time())
a.set_step_size(a.get_step_size())
a.set_start_time(a.get_start_time())
a.set_end_time(a.get_end_time())
a.set_method(a.method)
print(a.get_rtol(), a.get_atol(), a.y)
a.show_system()

print("")

with de.BlockTimer(section_label="Integrator Tests") as sttimer:
    for i in sorted(de.available_methods):
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
            a.reset()
        except:
            raise RuntimeError("Test failed for integration method: {}".format(i))
    print("")
