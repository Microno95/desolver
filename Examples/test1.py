from __future__ import absolute_import, division, print_function, unicode_literals

import desolver as de
@de.rhs_prettifier("""[vx, x]""")
def rhs(t, state, **kwargs):
    x,vx = state

    dx  = vx
    dvx = -x

    return de.numpy.array([dx, dvx])

y_init = de.numpy.array([1., 0.])

a = de.OdeSystem(rhs, y0=y_init, n=y_init.shape, t=(0, 2*de.numpy.pi), stpsz=0.01, rtol=1e-1, atol=1e-3)

a.set_end_time(a.get_end_time())
a.set_step_size(a.get_step_size())
a.set_start_time(a.get_start_time())
a.set_current_time(a.get_current_time())
a.set_method(a.get_method())
# a.set_dimensions(a.get_dimensions())
a.record_trajectory(True)
print(a.get_rtol(), a.get_atol(), a.get_trajectory())
a.show_system()

print("")

with de.BlockTimer(section_label="Integrator Tests") as sttimer:
    for i in de.OdeSystem.available_methods(suppress_print=True):
        print("Testing {}".format(str(i)))
        try:
            a.set_method(i)
            a.integrate()
            max_diff = de.numpy.max(de.numpy.abs(a.get_trajectory()[0] - a.get_trajectory()[-1]).ravel())
            if de.OdeSystem.available_methods(suppress_print=True)[i].__adaptive__:
                assert(max_diff < a.atol * 10)
            print("{} Succeeded with max_diff = {}".format(str(i), max_diff))
            print()
            a.reset()
        except:
            raise RuntimeError("Test failed for integration method: {}".format(i))
    print("")
