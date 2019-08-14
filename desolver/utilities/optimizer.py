import numpy
from .. import backend as D

def brentsroot(f, lower_bound, upper_bound, tol=None, verbose=False):
    """Brent's algorithm for finding root of a bracketed function.

    Parameters
    ----------
    f : callable
        callable that evaluates the function whose roots are to be found
    lower_bound : float-type
        lower bound of interval to find root in
    upper_bound : float-type
        upper bound of interval to find root in
    tol : float-type
        numerical tolerance for the precision of the root
    verbose : bool
        set to true to print useful information

    Returns
    -------
    float-type or None
        returns None if a root is not found, otherwise returns the root of the function

    Examples
    --------
    ```python
    >>> def ft(x):
        return x**2 - (1 - x)**5
    >>> xl, xu = 0.1, 1.0
    >>> x0 = D.brentsroot(ft, xl, xu, verbose=True)
    >>> x0, ft(x0)
    (0.34595481584824206, 6.938893903907228e-17)
    ```
    """
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
    return b

def brentsrootvec(flist, lower_bound, upper_bound, tol=None, verbose=False):
    if hasattr(flist, '__len__'):
        try:
            flen = len(flist)
        except:
            flen = 0
    else:
        flen = 0
    if hasattr(lower_bound, '__len__'):
        try:
            lblen = len(lower_bound)
        except:
            lblen = 0
    else:
        lblen = 0
    if hasattr(upper_bound, '__len__'):
        try:
            ublen = len(upper_bound)
        except:
            ublen = 0
    else:
        ublen = 0
        
    if flen == lblen == ublen and flen > 0:
        return D.stack([
            brentsroot(flist[i], lower_bound[i], upper_bound[i], tol=tol, verbose=verbose) for i in range(flen)
        ])
    elif flen == lblen and flen > 0 and ublen == 0:
        return D.stack([
            brentsroot(flist[i], lower_bound[i], upper_bound, tol=tol, verbose=verbose) for i in range(flen)
        ])
    elif flen == ublen and lblen == 0 and flen > 0:
        return D.stack([
            brentsroot(flist[i], lower_bound, upper_bound[i], tol=tol, verbose=verbose) for i in range(flen)
        ])
    elif lblen == ublen and lblen > 0 and flen == 0:
        return D.stack([
            brentsroot(flist, lower_bound[i], upper_bound[i], tol=tol, verbose=verbose) for i in range(lblen)
        ])
    elif flen == lblen == 0 and ublen > 0:
        return D.stack([
            brentsroot(flist, lower_bound, upper_bound[i], tol=tol, verbose=verbose) for i in range(ublen)
        ])
    elif flen == ublen == 0 and lblen > 0:
        return D.stack([
            brentsroot(flist, lower_bound[i], upper_bound, tol=tol, verbose=verbose) for i in range(lblen)
        ])
    elif lblen == ublen == 0 and flen > 0:
        return D.stack([
            brentsroot(flist[i], lower_bound, upper_bound, tol=tol, verbose=verbose) for i in range(flen)
        ])
    else:
        return brentsroot(flist, lower_bound, upper_bound, tol=tol, verbose=verbose)
    
brentsrootvec.__doc__ = "Vectorized" + brentsroot.__doc__

# def brentsrootvec(f, lower_bound, upper_bound, tol=None, verbose=False):
#     if tol is None:
#         tol = D.epsilon()
#     if tol < D.epsilon():
#         tol = D.epsilon()
#     tol = D.to_float(tol)
#     a,b = lower_bound, upper_bound
#     conv  = D.ones_like(a, dtype=bool)
#     fa = f(a)
#     fb = f(b)
    
#     mask = (D.abs(fa) < D.abs(fb))
#     a[mask],  b[mask]  = b[mask],  a[mask]
#     fa[mask], fb[mask] = fb[mask], fa[mask]
    
#     c  = D.copy(a)
#     d  = D.copy(b)
#     s  = D.copy(a)
#     fc = f(c)
#     fs = D.copy(fc)
    
#     mflag = D.ones_like(a, dtype=bool)
#     conv[fa * fb >= 0] = False
#     not_conv = D.logical_not(conv)
#     numiter = D.ones_like(a, dtype=D.int64)*3
    
#     while D.any(conv):
#         if verbose:
#             print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}, conv={not_conv}".format(**locals()))
#         mask = D.logical_and(fa != fc, fb != fc)
#         mask[D.logical_not(conv)] = False 
#         s[mask] = (a[mask] * fb[mask] * fc[mask]) / ((fa[mask] - fb[mask])*(fa[mask] - fc[mask]))
#         s[mask] = s[mask] + (b[mask] * fa[mask] * fc[mask]) / ((fb[mask] - fa[mask])*(fb[mask] - fc[mask]))
#         s[mask] = s[mask] + (c[mask] * fa[mask] * fb[mask]) / ((fc[mask] - fa[mask])*(fc[mask] - fb[mask]))
#         mask = D.logical_not(mask)
#         mask[D.logical_not(conv)] = False
#         s[mask] = b[mask] - fb[mask] * (b[mask] - a[mask]) / (fb[mask] - fa[mask])
            
#         cond1 = D.logical_not(D.logical_or(D.logical_and((3 * a + b) / 4 < s, s < b), D.logical_and(b < s, s < (3 * a + b) / 4)))
#         mask  = cond1
#         cond2 = D.logical_and(mflag, D.abs(s - b) >= D.abs(b - c)/2)
#         mask  = D.logical_or(mask, cond2)
#         cond3 = D.logical_and(D.logical_not(mflag), D.abs(s - b) >= D.abs(c - d) / 2)
#         mask  = D.logical_or(mask, cond3)
#         cond4 = D.logical_and(mflag, D.abs(b - c) < tol)
#         mask  = D.logical_or(mask, cond4)
#         cond5 = D.logical_and(D.logical_not(mflag), D.abs(c - d) < tol)
#         mask  = D.logical_or(mask, cond5)
#         mask[D.logical_not(conv)] = False
#         s[mask] = (a[mask] + b[mask]) / 2
#         mflag[mask] = True
#         mask  = D.logical_not(mask)
#         mask[D.logical_not(conv)] = False
#         mflag[mask] = False
            
#         fs = f(s)
#         numiter[conv] = numiter[conv] + 1
#         d  = c
        
#         mask = fa * fs < 0
#         mask[D.logical_not(conv)] = False
#         b[mask]  = s[mask]
#         fb[mask] = fs[mask]
#         mask = D.logical_not(mask)
#         mask[D.logical_not(conv)] = False
#         a[mask]  = s[mask]
#         fa[mask] = fs[mask]        
    
#         mask = D.abs(fa) < D.abs(fb)
#         mask[D.logical_not(conv)] = False
#         a[mask],  b[mask]  = b[mask],  a[mask]
#         fa[mask], fb[mask] = fb[mask], fa[mask]
        
#         conv = D.logical_not(D.logical_or(D.logical_or(fb == 0, fs == 0), D.abs(b - a) < tol))
#         not_conv = D.logical_not(conv)
#     if verbose:
#         print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}, conv={not_conv}".format(**locals()))
#     return b