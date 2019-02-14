from __future__ import absolute_import, division, print_function, unicode_literals

import desolver as de

@de.rhs_prettifier("""[vx, x]""")
def rhs(t, state, **kwargs):
    x,vx = state

    dx  = vx
    dvx = -x + t

    return de.numpy.array([dx, dvx])

def analytic_soln(t, initial_conditions):
    c1 =  initial_conditions[0]
    c2 =  initial_conditions[1] - 1.0
    return de.numpy.array([
        c2 * de.numpy.sin(t) + c1 * de.numpy.cos(t) + t,
        c2 * de.numpy.cos(t) - c1 * de.numpy.sin(t) + 1.
    ])

def kbinterrupt_cb(ode_sys):
    if ode_sys.t[-1] > de.numpy.pi:
        raise KeyboardInterrupt("Test Interruption and Catching")

y_init = de.numpy.array([1., 0.])

a = de.OdeSystem(rhs, y0=y_init, n=y_init.shape, dense_output=True, t=(0, 2*de.numpy.pi), dt=0.01, rtol=1e-6, atol=1e-9)

a.set_end_time(a.get_end_time())
a.set_step_size(a.get_step_size())
a.set_start_time(a.get_start_time())
a.set_end_time(a.get_end_time())
a.set_method(a.method)
a.set_dimensions(a.dim)
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

            max_diff = de.numpy.max(de.numpy.abs(analytic_soln(a.t[-1], a.y[0])-a.y[-1]))
            if a.method.__adaptive__:
                assert(max_diff < a.atol * 10)
            print("{} Succeeded with max_diff from analytical solution = {}".format(str(i), max_diff))
            print()
            a.reset()
        except:
            raise RuntimeError("Test failed for integration method: {}".format(i))
    print("")
