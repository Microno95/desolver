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

import time
import sys

__all__ = [
    'convert_suffix',
    'warning',
    'search_bisection',
    'BlockTimer'
]

def convert_suffix(value, suffixes=('d', 'h', 'm', 's'), ratios=(24, 60, 60), delimiter=':'):
    """Converts a base value into a human readable format with the given suffixes and ratios.

    Parameters
    ----------
    value : int or float
        value to be converted
    suffixes : list of str
        suffixes for each subdivision (eg. ['days', 'hours', 'minutes', 'seconds'])
    ratios : list of int
        the relative period of each subdivision (eg. 24 hours in 1 day, 60 minutes in 1 hour, 
        60 seconds in 1 minute -> [24, 60, 60])
    delimiter : str
        string to use between each subdivision

    Returns
    -------
    str
        returns string with the subdivision values and suffixes joined together

    Examples
    --------
    ```python
    >>> convert_suffix(3661, suffixes=['d', 'h', 'm', 's'], ratios=[24, 60, 60], delimiter=':')
    '0d:1h:1m1.00s'
    ```
    """
    tValue = value
    outputValues = []
    for i in ratios[::-1]:
        outputValues.append(int(tValue % i))
        tValue = (tValue - tValue % i) // i
    outputValues.append(tValue)
    ret_string = delimiter.join(["{}{}".format(int(i[0]), i[1]) for i in zip(outputValues[::-1][:-1], suffixes[:-1])])
    ret_string = ret_string + "{:.2f}{}".format(outputValues[0], suffixes[-1])
    return ret_string

def warning(*args, **kwargs):
    """Convenience function for printing to sys.stderr. 

    Parameters
    ----------
    args : variable
        arguments to be passed to print
    kwargs : variable
        keyword arguments to be passed to print
        
    Examples
    --------
    ```python
    >>> warning("Things have failed...")
    Things have failed...
    ```
    """
    print(*args, file=sys.stderr, **kwargs)

def search_bisection(array, val):
    """Finds the index of the nearest value to val in array. Uses the bisection method.

    Parameters
    ----------
    array : list of numeric values
        list to search, assumes the list is sorted (will not work if it isn't sorted!)
    val : numeric
        numeric value to find the nearest value in the array.

    Returns
    -------
    int
        returns the index of the position in the array with the value closest to val

    Examples
    --------
    ```python
    >>> list_to_search = [1,2,3,4,5]
    >>> val_to_find    = 2.5
    >>> idx = search_bisection(list_to_search, val_to_find)
    >>> idx, list_to_search[idx]
    (1, 2)
    ```
    """
    
    jlower = 0
    jupper = len(array) - 1
    
    if val <= array[jlower]:
        return jlower
    elif val >= array[jupper]:
        return jupper
    
    while (jupper - jlower) > 1:
        jmid = (jupper + jlower) // 2
        if (val >= array[jmid]):
            jlower = jmid
        else:
            jupper = jmid
            
    return jlower

class BlockTimer():
    """Timing Class

    Takes advantage of the with syntax in order to time a block of code.
    
    Parameters
    ----------
    section_label : str
        name given to section of code
    start_now : bool
        if True the timer is started upon construction, otherwise start() must be called.
    suppress_print : bool
        if True a message will be printed upon destruction with the section_label and the code time.
    """
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
        """Method to start the timer
        """
        self.start_time = time.perf_counter()

    def end(self):
        """Method to stop the timer
        """
        self.end_time = time.perf_counter()
        self.stopped  = True

    def elapsed(self):
        """Method to get the elapsed time.
        
        Returns
        -------
        float
            Returns the elapsed time since timer start if stop() was not called, otherwise returns the elapsed time
            between timer start and when elapsed() is called.        
        """
        
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        else:
            return self.end_time - self.start_time
        
    def restart_timer(self):
        """Method to restart the timer. 
        
        Sets the start time to now and resets the end time.
        """
        if self.stopped:
            self.stopped = False
        self.start_time = time.perf_counter()
        self.end_time   = None
