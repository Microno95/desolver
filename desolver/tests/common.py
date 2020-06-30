import desolver as de
import desolver.backend as D

def set_up_basic_system():

    de_mat = D.array([[0.0, 1.0], [-1.0, 0.0]])

    @de.rhs_prettifier("""[vx, -x+t]""")
    def rhs(t, state, **kwargs):
        return de_mat @ state + D.array([0.0, t])

    def analytic_soln(t, initial_conditions):
        c1 = initial_conditions[0]
        c2 = initial_conditions[1] - 1

        return D.stack([
            c2 * D.sin(D.to_float(D.asarray(t))) + c1 * D.cos(D.to_float(D.asarray(t))) + D.asarray(t),
            c2 * D.cos(D.to_float(D.asarray(t))) - c1 * D.sin(D.to_float(D.asarray(t))) + 1
        ])

    y_init = D.array([1., 0.])

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon() ** 0.5,
                     atol=D.epsilon() ** 0.5, constants=dict(k=1.0))

    return de_mat, rhs, analytic_soln, y_init, a