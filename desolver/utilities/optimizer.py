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

import numpy
from .. import backend as D

__all__ = [
    'brentsroot',
    'brentsrootvec'
]

def brentsroot(f, bounds, tol=None, verbose=False):
    """Brent's algorithm for finding root of a bracketed function.

    Parameters
    ----------
    f : callable
        callable that evaluates the function whose roots are to be found
    bounds : tuple of float, shape (2,)
        lower and upper bound of interval to find root in
    tol : float-type
        numerical tolerance for the precision of the root
    verbose : bool
        set to true to print useful information

    Returns
    -------
    tuple(float-type, bool)
        returns the location of the root if found and a bool indicating a root was found

    Examples
    --------
    ```python
    >>> def ft(x):
        return x**2 - (1 - x)**5
    >>> xl, xu = 0.1, 1.0
    >>> x0, success = brentsroot(ft, xl, xu, verbose=True)
    >>> success, x0, ft(x0)
    (True, 0.34595481584824206, 6.938893903907228e-17)
    ```
    """
    lower_bound, upper_bound = bounds
    if tol is None:
        tol = D.epsilon()
    if tol < D.epsilon():
        tol = D.epsilon()
    tol = D.to_float(tol)
    a,b = D.to_float(lower_bound), D.to_float(upper_bound)
    fa = f(a)
    fb = f(b)
    
    if fa*fb >= 0:
        return D.to_float(numpy.inf)
    if D.abs(fa) < D.abs(fb):
        a,b = b,a
        fa,fb = fb,fa
    
    c  = D.copy(a)
    d  = D.copy(b)
    fc = f(c)
    
    mflag = True
    conv  = False
    numiter = 3
    
    while not conv:
        if verbose:
            print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}".format(**locals()))
        if fa != fc and fb != fc:
            s =     (a * fb * fc) / ((fa - fb)*(fa - fc))
            s = s + (b * fa * fc) / ((fb - fa)*(fb - fc))
            s = s + (c * fa * fb) / ((fc - fa)*(fc - fb))
        else:
            s = b - fb * (b - a) / (fb - fa)
            
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        bisect_now = cond1
        if bisect_now:
            mflag = True
        else:
            cond2 = mflag and D.abs(s - b) >= D.abs(b - c)/2
            bisect_now = bisect_now or cond2
        if bisect_now:
            mflag = True
        else:
            cond3 = not mflag and D.abs(s - b) >= D.abs(c - d) / 2
            bisect_now = bisect_now or cond3
        if bisect_now:
            mflag = True
        else:
            cond4 = mflag and D.abs(b - c) < tol
            bisect_now = bisect_now or cond4
        if bisect_now:
            mflag = True
        else:
            cond5 = not mflag and D.abs(c - d) < tol
            bisect_now = bisect_now or cond5
        if bisect_now:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False
            
        fs = f(s)
        numiter += 1
        d  = c
        
        if fa * fs < 0:
            b  = s
            fb = fs
        else:
            a  = s
            fa = fs
        
        if D.abs(fa) < D.abs(fb):
            a,b = b,a
            fa,fb = fb,fa
        
        conv = (fb == 0 or fs == 0 or D.abs(b - a) < tol)
    if verbose:
        print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}".format(**locals()))
    return b, D.abs(f(b)) <= tol

# def brentsrootvec(flist, lower_bound, upper_bound, tol=None, verbose=False):
#     """Vectorized Brent's algorithm for finding root of a bracketed function.

#     Parameters
#     ----------
#     f : callable or list of callables
#         function or list of functions whose roots need to be found
#     lower_bound : float-type or list of float-types
#         lower bound(s) of intervals(s) to find the root in
#     upper_bound : float-type or list of float-type
#         upper bound(s) of intervals(s) to find the root in
#     tol : float-type
#         numerical tolerance for the precision of the root
#     verbose : bool
#         set to true to print useful information

#     Returns
#     -------
#     float-type or None
#         returns None if a root is not found, otherwise returns the root of the function

#     Examples
#     --------
#     ```python
#     >>> def ft(x):
#         return x**2 - (1 - x)**5
#     >>> xl, xu = 0.1, 1.0
#     >>> x0 = D.brentsroot(ft, xl, xu, verbose=True)
#     >>> x0, ft(x0)
#     (0.34595481584824206, 6.938893903907228e-17)
#     ```
#     """
    
#     if hasattr(flist, '__len__'):
#         try:
#             flen = len(flist)
#         except:
#             flen = 0
#     else:
#         flen = 0
        
#     if hasattr(lower_bound, '__len__'):
#         try:
#             lblen = len(lower_bound)
#         except:
#             lblen = 0
#     else:
#         lblen = 0
#     if hasattr(upper_bound, '__len__'):
#         try:
#             ublen = len(upper_bound)
#         except:
#             ublen = 0
#     else:
#         ublen = 0
        
#     if flen == lblen == ublen and flen > 0:
#         return D.stack([
#             brentsroot(flist[i], lower_bound[i], upper_bound[i], tol=tol, verbose=verbose) for i in range(flen)
#         ])
#     elif flen == lblen and flen > 0 and ublen == 0:
#         return D.stack([
#             brentsroot(flist[i], lower_bound[i], upper_bound, tol=tol, verbose=verbose) for i in range(flen)
#         ])
#     elif flen == ublen and lblen == 0 and flen > 0:
#         return D.stack([
#             brentsroot(flist[i], lower_bound, upper_bound[i], tol=tol, verbose=verbose) for i in range(flen)
#         ])
#     elif lblen == ublen and lblen > 0 and flen == 0:
#         return D.stack([
#             brentsroot(flist, lower_bound[i], upper_bound[i], tol=tol, verbose=verbose) for i in range(lblen)
#         ])
#     elif flen == lblen == 0 and ublen > 0:
#         return D.stack([
#             brentsroot(flist, lower_bound, upper_bound[i], tol=tol, verbose=verbose) for i in range(ublen)
#         ])
#     elif flen == ublen == 0 and lblen > 0:
#         return D.stack([
#             brentsroot(flist, lower_bound[i], upper_bound, tol=tol, verbose=verbose) for i in range(lblen)
#         ])
#     elif lblen == ublen == 0 and flen > 0:
#         return D.stack([
#             brentsroot(flist[i], lower_bound, upper_bound, tol=tol, verbose=verbose) for i in range(flen)
#         ])
#     else:
#         return brentsroot(flist, lower_bound, upper_bound, tol=tol, verbose=verbose)

def brentsrootvec(f, bounds, tol=None, verbose=False):
    """Vectorised Brent's algorithm for finding root of bracketed functions.

    Parameters
    ----------
    f : list of callables
        list of callables each of which evaluates the function to find the root of
    bounds : tuple of float, shape (2,)
        lower and upper bound of interval to find root in
    tol : float-type
        numerical tolerance for the precision of the roots
    verbose : bool
        set to true to print useful information

    Returns
    -------
    tuple(list(float-type), list(bool))
        returns a list of the locations of roots and a list of bools indicating whether or not a root was found in the interval

    Examples
    --------
    ```python
    >>> f = lambda x: lambda y: x * y - y**2 + x
    >>> xl, xu = 0.1, 1.0
    >>> funcs = [f(i*0.5) for i in range(3)]
    >>> x0, success = brentsrootvec(funcs, xl, xu, verbose=True)
    >>> success, x0, [funcs[i](x0[i]) for i in range(len(funcs))]
    (array([ True,  True,  True]), array([0.        , 1.        , 1.61803399]), [0.0, 0.0, 0.0])
    ```
    """
    lower_bound, upper_bound = bounds
    if tol is None:
        tol = D.epsilon()
    if tol < D.epsilon():
        tol = D.epsilon()
    tol = D.to_float(tol)
    a,b = D.stack([lower_bound for _ in range(len(f))]), D.stack([upper_bound for _ in range(len(f))])
    
    def _f(x, msk=None):
        if msk is None:
            if verbose:
                print([f[i](x[i]) for i in range(len(f))])
                print(f[0](x[0]), f[1](x[1]))
            return D.stack([f[i](x[i]) for i in range(len(f))])
        else:
            return D.stack([f[i](x[i]) if msk[i] else D.to_float(0.0) for i in range(len(f))])
    
    if verbose:
        print(_f(a))
    
    conv = D.ones_like(a, dtype=bool)
    
    fa = _f(a)
    fb = _f(b)
    
    mask               = (D.abs(fa) < D.abs(fb))
    a[mask],  b[mask]  = b[mask],  a[mask]
    fa[mask], fb[mask] = fb[mask], fa[mask]
    
    c  = D.copy(a)
    d  = D.copy(b)
    s  = D.copy(a)
    fc = _f(c)
    fs = D.copy(fc)
    
    mflag              = D.ones_like(a, dtype=bool)
    conv[fa * fb >= 0] = False
    not_conv           = D.logical_not(conv)
    numiter            = D.ones_like(a, dtype=D.int64)*3
    true_conv          = D.abs(_f(b)) <= tol
    
    while D.any(conv):
        if verbose:
            print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}, conv={not_conv}".format(**locals()))
        mask                      = D.logical_and(fa != fc, fb != fc)
        mask[not_conv]            = False 
        s[mask]                   = (a[mask] * fb[mask] * fc[mask]) / ((fa[mask] - fb[mask])*(fa[mask] - fc[mask]))
        s[mask]                   = s[mask] + (b[mask] * fa[mask] * fc[mask]) / ((fb[mask] - fa[mask])*(fb[mask] - fc[mask]))
        s[mask]                   = s[mask] + (c[mask] * fa[mask] * fb[mask]) / ((fc[mask] - fa[mask])*(fc[mask] - fb[mask]))
        mask                      = D.logical_not(mask)
        mask[D.logical_not(conv)] = False
        s[mask]                   = b[mask] - fb[mask] * (b[mask] - a[mask]) / (fb[mask] - fa[mask])
            
        cond1          = D.logical_not(D.logical_or(D.logical_and((3 * a + b) / 4 < s, s < b), D.logical_and(b < s, s < (3 * a + b) / 4)))
        mask           = cond1
        cond2          = D.logical_and(mflag, D.abs(s - b) >= D.abs(b - c)/2)
        mask           = D.logical_or(mask, cond2)
        cond3          = D.logical_and(D.logical_not(mflag), D.abs(s - b) >= D.abs(c - d) / 2)
        mask           = D.logical_or(mask, cond3)
        cond4          = D.logical_and(mflag, D.abs(b - c) < tol)
        mask           = D.logical_or(mask, cond4)
        cond5          = D.logical_and(D.logical_not(mflag), D.abs(c - d) < tol)
        mask           = D.logical_or(mask, cond5)
        mask[not_conv] = False
        s[mask]        = (a[mask] + b[mask]) / 2
        mflag[mask]    = True
        mask           = D.logical_not(mask)
        mask[not_conv] = False
        mflag[mask]    = False
            
        fs                 = _f(s, conv)
        numiter[conv]      = numiter[conv] + 1
        d                  = c
        
        mask               = fa * fs < 0
        mask[not_conv]     = False
        b[mask]            = s[mask]
        fb[mask]           = fs[mask]
        mask               = D.logical_not(mask)
        mask[not_conv]     = False
        a[mask]            = s[mask]
        fa[mask]           = fs[mask]        
    
        mask               = D.abs(fa) < D.abs(fb)
        mask[not_conv]     = False
        a[mask],  b[mask]  = b[mask],  a[mask]
        fa[mask], fb[mask] = fb[mask], fa[mask]
        
        conv               = D.logical_not(D.logical_or(D.logical_or(fb == 0, fs == 0), D.abs(b - a) < tol))
        not_conv           = D.logical_not(conv)
        true_conv          = D.abs(_f(b)) <= tol
        
        if D.any(numiter > 1000):
            break
    if verbose:
        print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}, conv={true_conv}".format(**locals()))
    return b, true_conv