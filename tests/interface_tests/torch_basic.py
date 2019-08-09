from __future__ import absolute_import, division, print_function, unicode_literals

import os

os.environ['DES_BACKEND']          = 'torch'
os.environ['CUDA_DEVICE_ORDER']    = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import desolver as de
import desolver.backend as D

device = D.torch.device("cuda:0" if D.torch.cuda.is_available() else "cpu")
D.torch.set_num_threads(1)

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

    y_init = D.array([1., 0.], device=device)

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, 2*D.pi), dt=0.01, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5)

    with de.BlockTimer(section_label="Integrator Tests") as sttimer:
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
            except:
                raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
        print("")

    print("{} backend test passed successfully!".format(os.environ['DES_BACKEND']))