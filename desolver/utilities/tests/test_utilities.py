import desolver as de
import desolver.backend as D
import numpy as np


def test_convert_suffix():
    assert (de.utilities.convert_suffix(3661) == "0d:1h:1m1.00s")


def test_bisection_search():
    l1 = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]

    for idx, i in enumerate(l1):
        assert (de.utilities.search_bisection(l1, i) == idx)


def test_blocktimer():
    with de.utilities.BlockTimer(start_now=False) as test:
        assert (test.start_time is None and not test.start_now)

    with de.utilities.BlockTimer(start_now=False) as test:
        assert (isinstance(test.start_now, bool) and test.start_now == False)
        assert (test.start_time is None)
        assert (test.end_time is None)
        test.start()
        assert (isinstance(test.start_time, float))
        assert (isinstance(test.elapsed(), float) and test.elapsed() > 0)
        test.end()
        assert (isinstance(test.end_time, float))
        assert (isinstance(test.elapsed(), float) and test.elapsed() > 0)
        assert (test.stopped == True)
        test.restart_timer()
        assert (test.end_time is None)
        assert (test.stopped == False)


if __name__ == "__main__":
    np.testing.run_module_suite()
