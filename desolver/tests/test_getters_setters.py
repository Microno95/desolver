import desolver as de
import desolver.backend as D
import numpy as np

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

        assert(isinstance(a.integrator, de.available_methods(False)["RK45CK"]))

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
        
        a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5)
        
        a.integrate()

        try:
            print(str(a))
            print(repr(a))
            assert(D.max(D.abs(a.sol(a.t[0]) - y_init)) <= 8*D.epsilon()**0.5)
            assert(D.max(D.abs(a.sol(a.t[-1]) - analytic_soln(a.t[-1], y_init))) <= 8*D.epsilon()**0.5)
        except:
            raise
            
        
        print("{} backend test passed successfully!".format(D.backend()))
        
        
if __name__ == "__main__":
    np.testing.run_module_suite()
        