"""
The MIT License (MIT)

Copyright (c) 2017 Microno95, Ekin Ozturk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy
import time

from math import floor

def sa_minimisation(eqn_lambda, arg_name=None, start=0.0, eqn_args=None, eqn_kwargs=None, init_temperature=1.0, iterlimit=12800, tol=1e-4):
    eqn_args = eqn_args if eqn_args is not None else tuple()
    eqn_kwargs = eqn_kwargs if eqn_kwargs is not None else dict()
    arg_name = [arg_name] if isinstance(arg_name, str) else list(arg_name) if not isinstance(arg_name, list) and arg_name is not None else arg_name
    if arg_name is not None and isinstance(arg_name[0], str):
        def wrapped_eqn(x):
            return eqn_lambda(*eqn_args, **dict([(arg_name[i], x[i]) for i in range(len(arg_name))] + [(k, v) for k, v in eqn_kwargs.items()]))
    else:
        def wrapped_eqn(x):
            return eqn_lambda(*tuple(x + eqn_args), **eqn_kwargs)
    current_value = start
    proposed_value = start
    evaluated_value = wrapped_eqn(current_value)
    temperature = init_temperature
    for k in range(iterlimit):
        if temperature < tol:
            break
        proposed_value = current_value + numpy.random.randn(*numpy.shape(current_value))
        evaluated_proposed_value = wrapped_eqn(proposed_value)
        delta_E = evaluated_proposed_value - evaluated_value
        if (delta_E <= 0) or numpy.exp(-delta_E / temperature) >= numpy.random.uniform():
            current_value = proposed_value
            evaluated_value = evaluated_proposed_value
        temperature = temperature * 0.95 ** floor(k / 1000)
    return current_value

def extrap(x, xdata, ydata):
    """
    Richardson Extrapolation.

    Calculates the extrapolated values of a function evaluated at xdata
    with the values ydata.

    Required Arguments:
    x : The value(s) to extrapolate to (can be a numpy array)
    xdata, ydata : Values at which a function is evaluated, the result of that evaluation
    """
    coeff = []
    xlen = len(xdata)
    coeff.append([0, ydata[0]])
    for j in range(1, xlen):
        try:
            coeff.append([0, ydata[j]])
            for k in range(2, j + 1):
                if numpy.all([(i < 5e-14) for i in numpy.abs(coeff[-2][-1] - coeff[-1][-1])]):
                    raise StopIteration
                t1 = xdata[j] - xdata[j - k + 1]
                s1 = (x - xdata[j - k + 1])
                s2 = (x - xdata[j])
                p1 = s1 * coeff[j][k - 1] - s2 * coeff[j - 1][k - 1]
                p1 /= t1
                coeff[j].append(p1)
        except StopIteration:
            coeff.pop()
            break
    return coeff

def convert_suffix(value=3661, suffixes=(' d', ' h', ' m', ' s'), ratios=(24, 60, 60), delimiter=':'):
    """
    Converts a base value into a human readable format with the given suffixes and ratios.
    """
    tValue = value
    outputValues = []
    for i in ratios[::-1]:
        outputValues.append(int(tValue % i))
        tValue = (tValue - tValue % i) // i
    outputValues.append(int(tValue))
    return delimiter.join(["{:2d}{}".format(*i) for i in zip(outputValues[::-1], suffixes)])

def warning(message):
    print(message)

def named_object(name, alt_names=tuple(), order=1.0, adaptive=False):
    def wrap(f):
        f.__name__ = str(name)
        f.__alt_names__ = alt_names
        f.__order__ = order
        f.__adaptive__ = adaptive
        return f
    return wrap

def resize_to_correct_dims(x, eqnum, dim):
    return numpy.resize(x, tuple([eqnum] + list(dim)))

class StateTimer():
    def __init__(self):
        self.start = time.perf_counter()
        self.stopped = 0
        self.stopped = False

    def stop_timer(self):
        self.stop = time.perf_counter()

    def get_elapsed_time(self):
        if not self.stopped:
            return time.perf_counter() - self.start
        else:
            return self.stop - self.start

    def continue_timer(self):
        if self.stopped:
            self.stopped = False

    def restart_timer(self):
        if self.stopped:
            self.stopped = False
        self.start = time.perf_counter()
