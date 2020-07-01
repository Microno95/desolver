import desolver as de
import desolver.backend as D
import numpy as np
import pytest


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator_name', sorted(set(de.available_methods(False).values()), key=lambda x: x.__name__))
def test_float_formats(ffmt, integrator_name):
    D.set_float_fmt(ffmt)

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(True)

    print("Testing {} float format".format(D.float_fmt()))

    from .common import set_up_basic_system

    de_mat, rhs, analytic_soln, y_init, _ = set_up_basic_system()

    def kbinterrupt_cb(ode_sys):
        if ode_sys[-1][0] > D.pi:
            raise KeyboardInterrupt("Test Interruption and Catching")

    y_init = D.array([1., 0.])

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2 * D.pi), dt=0.01, rtol=D.epsilon() ** 0.5,
                     atol=D.epsilon() ** 0.5)

    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        if "Heun-Euler" in integrator_name.__name__ and D.float_fmt() == "gdual_real128":
            print("skipping {} due to ridiculous timestep requirements.".format(integrator_name))
            return
        a.set_method(integrator_name)
        print("Testing {}".format(a.integrator))

        try:
            a.integrate(callback=kbinterrupt_cb, eta=False)
        except KeyboardInterrupt as e:
            pass

        a.integrate(eta=False)

        max_diff = D.max(D.abs(analytic_soln(a.t[-1], a.y[0]) - a.y[-1]))
        if a.method.__adaptive__ and max_diff >= a.atol * 10 + D.epsilon():
            print("{} Failed with max_diff from analytical solution = {}".format(a.integrator, max_diff))
            raise RuntimeError("Failed to meet tolerances for adaptive integrator {}".format(str(i)))
        else:
            print("{} Succeeded with max_diff from analytical solution = {}".format(a.integrator, max_diff))
        a.reset()
    print("")

    print("{} backend test passed successfully!".format(D.backend()))


if __name__ == "__main__":
    np.testing.run_module_suite()
