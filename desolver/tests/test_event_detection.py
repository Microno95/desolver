import desolver as de
import desolver.backend as D
import numpy as np
import pytest


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator_name', sorted(set(de.available_methods(False).values()), key=lambda x: x.__name__))
def test_event_detection_multiple(ffmt, integrator_name):
    if ffmt == 'float16':
        return
    D.set_float_fmt(ffmt)

    print("Testing event detection for float format {}".format(D.float_fmt()))

    from .common import set_up_basic_system

    de_mat, rhs, analytic_soln, y_init, _ = set_up_basic_system()

    def time_event(t, y, **kwargs):
        return t - D.pi / 8

    def second_time_event(t, y, **kwargs):
        return t - D.pi / 16

    time_event.is_terminal = True
    time_event.direction = 0
    second_time_event.is_terminal = False
    second_time_event.direction = 0

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi / 4), dt=0.01, rtol=D.epsilon() ** 0.5,
                     atol=D.epsilon() ** 0.5)

    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        try:
            a.set_method(integrator_name)
            print("Testing {}".format(a.integrator))
            assert (a.integration_status() == "Integration has not been run.")

            a.integrate(eta=False, events=[time_event, second_time_event])

            assert (a.integration_status() == "Integration terminated upon finding a triggered event.")

            try:
                assert (D.abs(a.t[-1] - D.pi / 8) <= 10 * D.epsilon())
                assert (D.abs(a.events[0].t - D.pi / 16) <= 10 * D.epsilon())
                assert (len(a.events) == 2)
            except:
                print("Event detection with integrator {} failed with t[-1] = {}".format(a.integrator, a.t[-1]))
                raise RuntimeError("Failed to detect event for integrator {}".format(str(i)))
            else:
                print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
            a.reset()
        except Exception as e:
            raise e
            # raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
    print("")

    print("{} backend test passed successfully!".format(D.backend()))


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator_name', sorted(set(de.available_methods(False).values()), key=lambda x: x.__name__))
def test_event_detection_single(ffmt, integrator_name):
    if ffmt == 'float16':
        return
    D.set_float_fmt(ffmt)

    print("Testing event detection for float format {}".format(D.float_fmt()))
    from .common import set_up_basic_system

    de_mat, rhs, analytic_soln, y_init, _ = set_up_basic_system()

    def time_event(t, y, **kwargs):
        return t - D.pi / 8

    time_event.is_terminal = True
    time_event.direction = 0

    a = de.OdeSystem(rhs, y0=y_init, dense_output=True, t=(0, D.pi / 4), dt=0.01, rtol=D.epsilon() ** 0.5,
                     atol=D.epsilon() ** 0.5)

    with de.utilities.BlockTimer(section_label="Integrator Tests") as sttimer:
        try:
            a.set_method(integrator_name)
            print("Testing {}".format(a.integrator))
            assert (a.integration_status() == "Integration has not been run.")

            a.integrate(eta=False, events=time_event)

            assert (a.integration_status() == "Integration terminated upon finding a triggered event.")

            try:
                assert (D.abs(a.t[-1] - D.pi / 8) <= 10 * D.epsilon())
                assert (len(a.events) == 1)
            except:
                print("Event detection with integrator {} failed with t[-1] = {}".format(a.integrator, a.t[-1]))
                raise RuntimeError("Failed to detect event for integrator {}".format(str(i)))
            else:
                print("Event detection with integrator {} succeeded with t[-1] = {}".format(a.integrator, a.t[-1]))
            a.reset()
        except Exception as e:
            raise e
            # raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
    print("")

    print("{} backend test passed successfully!".format(D.backend()))


if __name__ == "__main__":
    np.testing.run_module_suite()
