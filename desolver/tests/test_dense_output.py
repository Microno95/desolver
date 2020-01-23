import desolver as de
from desolver.differential_system import DenseOutput
import desolver.backend as D
import numpy as np

from nose.tools import *

def test_dense_init_and_call():
    denseoutput = DenseOutput(None, None)
    assert(denseoutput.t_eval == [0.0])
    assert(denseoutput.y_interpolants == [])
    
    
@raises(ValueError)
def test_dense_init_no_t():
    denseoutput = DenseOutput(None, [0.1])
    
@raises(ValueError)
def test_dense_init_no_y():
    denseoutput = DenseOutput([0.1], None)
    
@raises(ValueError)
def test_dense_init_mismatch_length():
    denseoutput = DenseOutput([0.1], [0.1, 0.1])
    
def test_dense_output():
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

        assert(D.max(D.abs(a[0].y - analytic_soln(a[0].t, y_init))) <= 4*D.epsilon())
        assert(D.max(D.abs(a[0].t)) <= 4*D.epsilon())
        assert(D.max(D.abs(a[-1].y - analytic_soln(a[-1].t, y_init))) <= 10*D.epsilon()**0.5)
        
        assert(D.max(D.abs(a[a[0].t].y - analytic_soln(a[0].t, y_init))) <= 4*D.epsilon())
        assert(D.max(D.abs(a[a[0].t].t)) <= 4*D.epsilon())
        assert(D.max(D.abs(a[a[-1].t].y - analytic_soln(a[-1].t, y_init))) <= 10*D.epsilon()**0.5)
        
        assert(D.max(D.abs(D.stack(a[a[0].t:a[-1].t].y) - D.stack(a.y))) <= 4*D.epsilon())
        assert(D.max(D.abs(D.stack(a[:a[-1].t].y) - D.stack(a.y))) <= 4*D.epsilon())
        
        assert(D.max(D.abs(D.stack(a[a[0].t:a[-1].t:2].y) - D.stack(a.y[::2]))) <= 4*D.epsilon())
        assert(D.max(D.abs(D.stack(a[a[0].t::2].y) - D.stack(a.y[::2]))) <= 4*D.epsilon())
        assert(D.max(D.abs(D.stack(a[:a[-1].t:2].y) - D.stack(a.y[::2]))) <= 4*D.epsilon())
    
if __name__ == "__main__":
    np.testing.run_module_suite()