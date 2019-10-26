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

import numpy

__all__ = [
    'e',
    'euler_gamma',
    'pi',
    'set_backend',
    'backend',
    'epsilon',
    'available_float_fmt',
    'float_fmt',
    'set_float_fmt',
    'cast_to_float_fmt'
]

_BACKEND      = 'numpy'
_FLOAT_FORMAT = 'float64'

# Constants
e           = 2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274
euler_gamma = 0.5772156649015328606065120900824024310421593359399235988057672348848677267776646709369470632917467495
pi          = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679


def set_backend(backend):
    """Method for setting the backend.

    Returns
    -------
    str
        The new backend as a string
    """
    global _BACKEND
    
    _BACKEND = str(backend)
    
def backend():
    """Method for determining the current backend.

    Returns
    -------
    str
        The current backend as a string

    Examples
    --------
    ```python
        >>> desolver.backend.backend()
        'numpy'
    ```
    """
    
    return _BACKEND

def epsilon():
    """Returns fuzz factor used in numeric expressions

    Returns
    -------
    float
        fuzz factor

    Example
    --------
    ```python
        >>> epsilon()
        '1e-07'
    ```
    """
    
    if _FLOAT_FORMAT == 'float16':
        return 5e-3
    elif _FLOAT_FORMAT == 'float32':
        return 5e-7
    elif _FLOAT_FORMAT == 'float64':
        return 5e-16
    
def available_float_fmt():
    if _BACKEND == 'numpy':
        return ['float16', 'float32', 'float64']
    else:
        return ['float32', 'float64']

def float_fmt():
    """Returns float format as a string

    Returns
    -------
    str
        str denoting the currently used float format

    See Also
    --------
    set_float_fmt: sets the float format

    Example
    --------
    ```python
        >>> float_format()
        'float64'
    ```
    """
    
    return _FLOAT_FORMAT

def set_float_fmt(new_fmt):
    """Sets the default float type

    Parameters
    ----------
    new_fmt : str
        New format for float type

    Raises
    ------
    ValueError
        When the float format is not one of the expected types or is incompatible with the backend

    See Also
    --------
    float_fmt: returns current default float type

    Examples
    --------
    ```python
        >>> from desolver import backend as D
        >>> D.float_format()
        'float64'
        >>> D.set_float_format('float16')
        >>> D.float_format()
        'float16'
    ```
    """
    
    global _FLOAT_FORMAT
    
    if _BACKEND == 'numpy':
        if new_fmt not in available_float_fmt():
            raise ValueError("Unknown float type " + str(new_fmt) + " for backend " + str(_BACKEND))
    elif _BACKEND == 'torch':
        import torch
        if new_fmt not in available_float_fmt():
            raise ValueError("Unknown float type " + str(new_fmt) + " for backend " + str(_BACKEND))
        elif new_fmt == 'float32':
            torch.set_default_dtype(torch.float32)
        elif new_fmt == 'float64':
            torch.set_default_dtype(torch.float64)
    
    _FLOAT_FORMAT = str(new_fmt)
    
def cast_to_float_fmt(x):
    """Cast a Numpy array to the default DESolver float type

    Parameters
    ----------
    x : np.ndarray
        Numpy array to be cast to DESolver float type

    Returns
    -------
    np.ndarray
        The same Numpy array cast to the same type as the DESolver float type

    See Also
    --------
    float_fmt: returns the current default float type
    set_float_fmt: sets the default float type

    Examples
    --------
    ```python
        >>> from desolver import backend as D
        >>> D.float_fmt()
        'float64'
        >>> arr = numpy.array([8.0, -1.0], dtype='float32')
        >>> arr.dtype
        dtype('float32')
        >>> new_arr = D.cast_to_float_fmt(arr)
        >>> new_arr
        array([ 8., -1.], dtype=float64)
        >>> new_arr.dtype
        dtype('float64')
    ```
    """
    
    return numpy.asarray(x, dtype=_FLOAT_FORMAT)