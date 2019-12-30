from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['DES_BACKEND']          = 'numpy'




import desolver as de
import desolver.backend as D
import numpy as np




def test_float_formats():
    for ffmt in D.available_float_fmt():
        D.set_float_fmt(ffmt)

        print("Testing {} float format".format(D.float_fmt()))

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
            if ode_sys[-1][0] > D.pi:
                raise KeyboardInterrupt("Test Interruption and Catching")

        y_init = D.array([1., 0.])

        a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5)

        with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
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
                except Exception as e:
                    print(e)
                    raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
            print("")

        print("{} backend test passed successfully!".format(os.environ['DES_BACKEND']))
        


def test_getter_setters():
    for ffmt in D.available_float_fmt():
        D.set_float_fmt(ffmt)

        print("Testing {} float format".format(D.float_fmt()))

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
            if ode_sys[-1][0] > D.pi:
                raise KeyboardInterrupt("Test Interruption and Catching")

        y_init = D.array([1., 0.])

        a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5)

        assert(a.get_end_time() == 2 * D.pi)
        assert(a.get_start_time() == 0)
        assert(a.dt == 0.01)
        assert(a.rtol == D.epsilon()**0.5)
        assert(a.atol == D.epsilon()**0.5)
        assert(D.norm(a.y[0]  - y_init) <= 2 * D.epsilon())
        assert(D.norm(a.y[-1] - y_init) <= 2 * D.epsilon())

        try:
            a.set_kick_vars([True, False])
        except Exception as e:
            raise RuntimeError("set_kick_vars failed with: {}".format(e))

        assert(a.staggered_mask == [True, False])
        pval = 3 * D.pi

        try:
            a.set_end_time(pval)
        except Exception as e:
            raise RuntimeError("set_end_time failed with: {}".format(e))

        assert(a.get_end_time() == pval)
        pval = -1.0

        try:
            a.set_start_time(pval)
        except Exception as e:
            raise RuntimeError("set_start_time failed with: {}".format(e))

        assert(a.get_start_time() == pval)
        assert(a.get_step_size() == 0.01)

        try:
            a.set_rtol(1e-3)
        except Exception as e:
            raise RuntimeError("set_rtol failed with: {}".format(e))

        assert(a.get_rtol() == 1e-3)

        try:
            a.set_atol(1e-3)
        except Exception as e:
            raise RuntimeError("set_atol failed with: {}".format(e))

        assert(a.get_atol() == 1e-3)

        try:
            a.set_method("RK45CK")
        except Exception as e:
            raise RuntimeError("set_method failed with: {}".format(e))

        assert(isinstance(a.integrator, de.available_methods["RK45CK"]))

        try:
            a.add_constants(k=5.0)
        except Exception as e:
            raise RuntimeError("add_constants failed with: {}".format(e))

        assert(a.consts['k'] == 5.0)

        try:
            a.remove_constants('k')
        except Exception as e:
            raise RuntimeError("remove_constants failed with: {}".format(e))

        assert('k' not in a.consts.keys())

        print("{} backend test passed successfully!".format(os.environ['DES_BACKEND']))
        
def test_event_detection():
    for ffmt in D.available_float_fmt():
        if ffmt == 'float16':
            continue
        D.set_float_fmt(ffmt)

        print("Testing event detection for float format {}".format(D.float_fmt()))

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
        
        y_init = D.array([1., 0.])

        def time_event(t, y, **kwargs):
            return t - D.pi/8
        
        time_event.is_terminal = True
        time_event.direction   = 0

        a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi/4), dt=0.01, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5)

        with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
            for i in sorted(set(de.available_methods.values()), key=lambda x:x.__name__):
                try:
                    a.set_method(i)
                    print("Testing {}".format(a.integrator))
                    a.integrate(eta=True, events=time_event)

                    if D.abs(a.t[-1] - D.pi/8) > 10*D.epsilon():
                        print("Event detection with integrator {} failed with t[-1] = {}".format(a.integrator, a.t[-1]))
                        raise RuntimeError("Failed to detect event for integrator {}".format(str(i)))
                    else:
                        print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
                    a.reset()
                except Exception as e:
                    raise e
                    raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
            print("")

        print("{} backend test passed successfully!".format(os.environ['DES_BACKEND']))
        
if __name__ == "__main__":
    np.testing.run_module_suite()