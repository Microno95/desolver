import desolver as de
import desolver.backend as D
import numpy as np

def test_event_detection_multiple():
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

        def second_time_event(t, y, **kwargs):
            return t - D.pi/16
        
        time_event.is_terminal = True
        time_event.direction   = 0
        second_time_event.is_terminal = False
        second_time_event.direction   = 0

        a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi/4), dt=0.01, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5)

        with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
            for i in sorted(set(de.available_methods(False).values()), key=lambda x:x.__name__):
                try:
                    a.set_method(i)
                    print("Testing {}".format(a.integrator))
                    assert(a.integration_status() == "Integration has not been run.")

                    a.integrate(eta=True, events=[time_event, second_time_event])
                    
                    assert(a.integration_status() == "Integration terminated upon finding a triggered event.")

                    try:
                        assert(D.abs(a.t[-1] - D.pi/8) <= 10*D.epsilon())
                        assert(D.abs(a.events[0].t - D.pi/16) <= 10*D.epsilon())
                        assert(len(a.events) == 2)
                    except:
                        print("Event detection with integrator {} failed with t[-1] = {}".format(a.integrator, a.t[-1]))
                        raise RuntimeError("Failed to detect event for integrator {}".format(str(i)))
                    else:
                        print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
                    a.reset()
                except Exception as e:
                    raise e
                    raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
            print("")

        print("{} backend test passed successfully!".format(D.backend()))
        
def test_event_detection_single():
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
            for i in sorted(set(de.available_methods(False).values()), key=lambda x:x.__name__):
                try:
                    a.set_method(i)
                    print("Testing {}".format(a.integrator))
                    assert(a.integration_status() == "Integration has not been run.")

                    a.integrate(eta=True, events=time_event)
                    
                    assert(a.integration_status() == "Integration terminated upon finding a triggered event.")

                    try:
                        assert(D.abs(a.t[-1] - D.pi/8) <= 10*D.epsilon())
                        assert(len(a.events) == 1)
                    except:
                        print("Event detection with integrator {} failed with t[-1] = {}".format(a.integrator, a.t[-1]))
                        raise RuntimeError("Failed to detect event for integrator {}".format(str(i)))
                    else:
                        print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
                    a.reset()
                except Exception as e:
                    raise e
                    raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
            print("")

        print("{} backend test passed successfully!".format(D.backend()))
        
        
if __name__ == "__main__":
    np.testing.run_module_suite()