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

        assert(a.t0 == 0)
        assert(a.tf == 2 * D.pi)
        assert(a.dt == 0.01)
        assert(a.rtol == D.epsilon()**0.5)
        assert(a.atol == D.epsilon()**0.5)
        assert(D.norm(a.y[0]  - y_init) <= 2 * D.epsilon())
        assert(D.norm(a.y[-1] - y_init) <= 2 * D.epsilon())

        a.set_kick_vars([True, False])

        assert(a.staggered_mask == [True, False])
        pval = 3 * D.pi

        a.tf = pval
    
        assert(a.tf == pval)
        pval = -1.0

        a.t0 = pval

        assert(a.t0 == pval)
        assert(a.dt == 0.01)

        a.rtol = 1e-3

        assert(a.rtol == 1e-3)

        a.atol = 1e-3

        assert(a.atol == 1e-3)

        a.set_method("RK45CK")

        assert(isinstance(a.integrator, de.available_methods(False)["RK45CK"]))

        a.add_constants(k=5.0)

        assert(a.consts['k'] == 5.0)

        a.remove_constants('k')

        assert('k' not in a.consts.keys())
        
def test_integration_and_representation():
    for ffmt in D.available_float_fmt():
        D.set_float_fmt(ffmt)

        print("Testing {} float format".format(D.float_fmt()))

        de_mat = D.array([[0.0, 1.0],[-1.0, 0.0]])

        @de.rhs_prettifier("""[vx, -x+t]""")
        def rhs(t, state, k, **kwargs):
            return de_mat @ state + D.array([0.0, t])

        def analytic_soln(t, initial_conditions):
            c1 = initial_conditions[0]
            c2 = initial_conditions[1] - 1
            
            return D.stack([
                c2 * D.sin(D.to_float(D.asarray(t))) + c1 * D.cos(D.to_float(D.asarray(t))) + D.asarray(t),
                c2 * D.cos(D.to_float(D.asarray(t))) - c1 * D.sin(D.to_float(D.asarray(t))) + 1
            ])

        y_init = D.array([1., 0.])

        a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5, constants=dict(k=1.0))
        
        assert(a.integration_status() == "Integration has not been run.")
        
        a.integrate()
        
        assert(a.integration_status() == "Integration completed successfully.")

        try:
            print(str(a))
            print(repr(a))
            assert(D.max(D.abs(a.sol(a.t[0]) - y_init)) <= 8*D.epsilon()**0.5)
            assert(D.max(D.abs(a.sol(a.t[-1]) - analytic_soln(a.t[-1], y_init))) <= 8*D.epsilon()**0.5)
            assert(D.max(D.abs(a.sol(a.t).T - analytic_soln(a.t, y_init))) <= 8*D.epsilon()**0.5)
        except:
            raise
            
        for i in a:
            assert(D.max(D.abs(i.y - analytic_soln(i.t, y_init))) <= 8*D.epsilon()**0.5)
            
        assert(len(a.y) == len(a))
        assert(len(a.t) == len(a))
        
def test_integration_and_nearestfloat_no_dense_output():
    for ffmt in D.available_float_fmt():
        D.set_float_fmt(ffmt)

        print("Testing {} float format".format(D.float_fmt()))

        de_mat = D.array([[0.0, 1.0],[-1.0, 0.0]])

        @de.rhs_prettifier("""[vx, -x+t]""")
        def rhs(t, state, k, **kwargs):
            return de_mat @ state + D.array([0.0, t])

        def analytic_soln(t, initial_conditions):
            c1 = initial_conditions[0]
            c2 = initial_conditions[1] - 1
            
            return D.stack([
                c2 * D.sin(D.to_float(D.asarray(t))) + c1 * D.cos(D.to_float(D.asarray(t))) + D.asarray(t),
                c2 * D.cos(D.to_float(D.asarray(t))) - c1 * D.sin(D.to_float(D.asarray(t))) + 1
            ])

        y_init = D.array([1., 0.])

        a = de.OdeSystem(rhs, y0=y_init, dense_output=False, t=(0, 2*D.pi), dt=0.01, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5, constants=dict(k=1.0))
        
        assert(a.integration_status() == "Integration has not been run.")
        
        a.integrate()
        
        assert(a.integration_status() == "Integration completed successfully.")
        
        assert(D.abs(a.t[-2] - a[2*D.pi].t) <= D.abs(a.dt))
        
if __name__ == "__main__":
    np.testing.run_module_suite()
        