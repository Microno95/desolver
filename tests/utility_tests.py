from __future__ import absolute_import, division, print_function, unicode_literals

import desolver as de
import desolver.backend as D

assert(de.deutil.convert_suffix() == "0.00d:1.00h:1.00m:1.00s")

l1 = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]

for idx, i in enumerate(l1):
    assert(de.utilities.search_bisection(l1, i) == idx)
    
with de.utilities.BlockTimer(start_now=False) as test:
    assert(isinstance(test.start_now, bool) and test.start_now == False)
    assert(test.start_time is None)
    assert(test.end_time is None)
    test.start()
    assert(isinstance(test.start_time, float))
    test.end()
    assert(isinstance(test.end_time, float))
    assert(isinstance(test.elapsed(), float) and test.elapsed() > 0)
    assert(test.stopped == True)
    test.restart_timer()
    assert(test.end_time is None)
    assert(test.stopped == False)