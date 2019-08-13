"""
The MIT License (MIT)

Copyright (c) 2019 Microno95, Ekin Ozturk

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

from .. import backend as D
import time
import sys

from math import floor

def convert_suffix(value=3661, suffixes=('d', 'h', 'm', 's'), ratios=(24, 60, 60), delimiter=':'):
    """
    Converts a base value into a human readable format with the given suffixes and ratios.
    """
    tValue = value
    outputValues = []
    for i in ratios[::-1]:
        outputValues.append(int(tValue % i))
        tValue = (tValue - tValue % i) // i
    outputValues.append(tValue)
    return delimiter.join(["{:.2f}{}".format(*i) for i in zip(outputValues[::-1], suffixes)])

def warning(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def search_bisection(array, value):
    jlower = 0
    jupper = len(array) - 1
    
    if value <= array[jlower]:
        return jlower
    elif value >= array[jupper]:
        return jupper
    
    while (jupper - jlower) > 1:
        jmid = (jupper + jlower) // 2
        if (value >= array[jmid]):
            jlower = jmid
        else:
            jupper = jmid
            
    return jlower

class BlockTimer():
    # Class to time a block of code.
    #
    # Designed to work using the with ... as ... : syntax.
    # Example:
    #       with block_timer(section_label="Test") as _timer:
    #           print(";".join(["{: >#5}".format(i-5) for i in range(21)]))
    #           print("Time elapsed since start: {}s".format(_timer.elapsed()))
    #
    def __init__(self, section_label=None, start_now=True, suppress_print=False):
        self.start_now = start_now
        self.stopped   = not start_now
        self.start_time = None
        self.end_time = None
        self.label = section_label
        self.suppress_print = suppress_print

    def __enter__(self):
        if self.start_now:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if not self.stopped:
            self.end()
        if not self.suppress_print:
            if self.start_time is None and not self.start_now:
                print("Timer was never started, cannot give run-time of code.")
                print("Set start_now=True to count from start of block to end")
            elif self.start_now:
                if self.label: print("Section:\n\t"+self.label)
                self.end_time = time.perf_counter()
                print("\tThis code block took {}".format(convert_suffix(self.end_time - self.start_time)))
            else:
                if self.label: print("Section:\n\t"+self.label)
                print("\tBetween start() and end(), the time taken was {}".format(convert_suffix(self.end_time - self.start_time)))

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def end(self):
        self.end_time = time.perf_counter()
        self.stopped  = True
        return self

    def elapsed(self):
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        else:
            return self.end_time - self.start_time
        
    def restart_timer(self):
        if self.stopped:
            self.stopped = False
        self.start_time = time.perf_counter()
        self.end_time   = None
