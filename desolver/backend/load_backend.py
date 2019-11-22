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

import os
import sys

from .common import *

if 'DES_BACKEND' in os.environ:
    set_backend(os.environ['DES_BACKEND'])
    assert(backend() == str(os.environ['DES_BACKEND']))

if backend() == 'numpy':
    from .numpy_backend import *
elif backend() == 'torch':
    from .torch_backend import *
else:
    raise ValueError("Unable to import backend : " + str(backend()))
    
print("Using " + str(backend()) + " backend", file=sys.stderr)

def contract_first_ndims(a, b, n=1):
    """Contracts the tensors a and b along the first n dimensions

    Simple interface to contract two tensors, a and b, along their first n dimensions.
    For example, for a (2,2,3,9) tensor and a (2,2,7) tensor, an n=1 contraction gives
    a (2,3,9,7) tensor and an n=2 contraction gives a (3,9,7) tensor.

    Parameters
    ----------
    a : array-type
        First tensor to contract with
    b : array-type
        Second tensor to contract with
    n : int
        The first indices to contract along

    Returns
    -------
    array-type
        The resultant tensor

    Raises
    ------
    ValueError
        If `n>len(shape(a))` ie. the contraction requires more dimensions than there exists

    See Also
    --------
    einsum: function used to write tensor operations via einstein notation

    Examples
    --------
    ```python
    >>> a = D.array([[0.0, 1.0],[1.0, 0.0]])
    >>> b = D.array([[2.0, 1.0],[3.0, 5.0]])
    >>> D.contract_first_ndims(a, b, n=1)
    array([3., 1.])
    >>> D.contract_first_ndims(a, b, n=2)
    4.0
    ```
    """
    if len(shape(a)) > len(shape(b)):
        a,b = b,a
    if n > len(shape(a)):
        raise ValueError("Cannot contract along more dims than there exists!")
    na = len(shape(a))
    nb = len(shape(b))
    einsum_str = "{},{}->{}"
    estr1      = "".join([chr(97 + i) for i in range(na)])
    estr2      = "".join([chr(97 + i) for i in range(nb)])
    estr3      = "".join([chr(97 + i + n) for i in range(nb - n)])
    einsum_str = einsum_str.format(estr1, estr2, estr3)
    return einsum(einsum_str, a, b)

del os, sys