from __future__ import absolute_import, division, print_function, unicode_literals

import desolver as de

a = de.OdeSystem(n=(3,3), equ=(("y_1*0.1", [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]),
                               ("-y_0*0.1", [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]),), t=(0, 10), relerr=1e-1)

a.set_end_time(a.get_end_time())
a.set_step_size(a.get_step_size())
a.set_start_time(a.get_start_time())
a.set_current_time(a.get_current_time())
a.set_method(a.get_method())
a.set_dimensions(a.get_dimensions())
a.record_trajectory(True)
print(a.get_relative_error(), a.get_trajectory())
a.show_system()

for i in de.OdeSystem.available_methods():
    try:
        a.set_method(i)
        a.set_step_size(1.0)
        a.integrate()
        a.reset()
    except:
        raise RuntimeError("Test failed for integration method: {}".format(i))
