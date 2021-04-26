import numpy
import scipy.optimize
from .. import backend as D
from . import utilities

__all__ = [
    'brentsroot',
    'brentsrootvec',
    'newtonraphson'
]

def brentsroot(f, bounds, tol=None, verbose=False, return_interval=False):
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
    
    >>> def ft(x):
        return x**2 - (1 - x)**5
    >>> xl, xu = 0.1, 1.0
    >>> x0, success = brentsroot(ft, xl, xu, verbose=True)
    >>> success, x0, ft(x0)
    (True, 0.34595481584824206, 6.938893903907228e-17)
    
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
    
    if fa*fb >= D.epsilon():
        return D.to_float(numpy.inf), False
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
            with numpy.printoptions(precision=17, linewidth=200):
                print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}".format(**locals()))
        if fa != fc and fb != fc:
            s =     (a * fb * fc) / ((fa - fb)*(fa - fc))
            s = s + (b * fa * fc) / ((fb - fa)*(fb - fc))
            s = s + (c * fa * fb) / ((fc - fa)*(fc - fb))
        else:
            s = b - fb * (b - a) / (fb - fa)
            
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = D.abs(s - b) >= D.abs(b - c)/2
        cond3 = D.abs(s - b) >= D.abs(c - d)/2
        cond4 = D.abs(b - c) < tol
        cond5 = D.abs(c - d) < tol
        bisect_now = cond1 or (mflag and cond2) or (not mflag and cond3) or (mflag and cond4) or (not mflag and cond5)
        mflag = bisect_now
        if mflag:
            s = (a + b) / 2

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
        if numiter >= 64:
            break
    if verbose:
        with numpy.printoptions(precision=17, linewidth=200):
            print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}".format(**locals()))
    if return_interval:
        return b, D.abs(f(b)) <= tol, (a, b)
    else:
        return b, D.abs(f(b)) <= tol

def brentsrootvec(f, bounds, tol=None, verbose=False, return_interval=False):
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
    
    >>> f = lambda x: lambda y: x * y - y**2 + x
    >>> xl, xu = 0.1, 1.0
    >>> funcs = [f(i*0.5) for i in range(3)]
    >>> x0, success = brentsrootvec(funcs, xl, xu, verbose=True)
    >>> success, x0, [funcs[i](x0[i]) for i in range(len(funcs))]
    (array([ True,  True,  True]), array([0.        , 1.        , 1.61803399]), [0.0, 0.0, 0.0])
    
    """
    lower_bound, upper_bound = bounds
    if tol is None:
        tol = D.epsilon()
    if tol < D.epsilon():
        tol = D.epsilon()
    tol = D.to_float(tol)
    a,b = D.stack([lower_bound for _ in range(len(f))]), D.stack([upper_bound for _ in range(len(f))])
    
    zero_elem = D.array(0.0)
    
    if D.backend() == 'torch':
        device = a.device
        zero_elem = zero_elem.to(device)
    
    def _f(x, msk=None):
        if msk is None:
            out = [f[i](x[i]) for i in range(len(f))]
        else:
            out = [f[i](x[i]) if msk[i] else zero_elem for i in range(len(f))]
        if verbose:
            print(msk)
            print(out)
            print(f[0](x[0]), f[1](x[1]))
        return D.stack(out)
    
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
            with numpy.printoptions(precision=17, linewidth=200):
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
        conv               = conv | (numiter >= 64)
        not_conv           = D.logical_not(conv)
        true_conv          = (D.abs(fb) <= tol)
        
    if verbose:
        with numpy.printoptions(precision=17, linewidth=200):
            print("[{numiter}] a={a}, b={b}, f(a)={fa}, f(b)={fb}, conv={true_conv}".format(**locals()))
    if return_interval:
        return b, true_conv, (a, b)
    else:
        return b, true_conv

def preconditioner(A, tol=None):
    if tol is None:
        if D.epsilon() <= 1e-5:
            tol = 32*D.epsilon()
        else:
            tol = D.epsilon()
    if tol < 32*D.epsilon() and D.epsilon() <= 1e-5:
        tol = 32*D.epsilon()
    I    = D.eye(A.shape[0])
    if D.backend() == 'torch':
        I = I.to(A)
    Pinv = D.zeros_like(A)
    A2   = A*A
    nA   = 0.5 * (D.sum(A2, axis=0)**0.5 + D.sum(A2, axis=1)**0.5)
    nA   = (nA > 32*D.epsilon())*nA + (nA <= 32*D.epsilon())
    Pinv = D.diag(nA)
        
    Ik = Pinv@A
    for _ in range(3):
        AP = A@Pinv
        Wn = -147*I + AP@(53*I + AP@(-11*I + AP))
        In0 = 0.75*Pinv + 0.25*0.25*Pinv@(32*I + AP@(-113*I + AP@(231*I + AP@(-301*I + AP@(259*I + AP@Wn)))))
        In1 = 2*Pinv - Ik@Pinv
        if D.norm(D.to_float(I - In0@A)) < D.norm(D.to_float(I - In1@A)):
            In = In0
        else:
            In = In1
        if D.norm(D.to_float(I - In@A)) >= D.norm(D.to_float(I - Ik)):
            break
        else:
            Pinv = In
        nPinv = D.norm(Pinv@A)
        Pinv  = Pinv / nPinv
        Ik    = Pinv@A
        if D.max(D.abs(D.to_float(Ik))) - 1 <= tol:
            break
    return Pinv

def estimate_cond(A):
    out = D.abs(A)
    out = out[out > 0]
    out = D.sqrt(D.max(out) / D.min(out))
    if out <= 32*D.epsilon():
        out = D.ones_like(out)
    return out

def newtonraphson(f, x0, jac=None, tol=None, verbose=False, maxiter=50, jac_update_rate=20, use_scipy=True):
    if tol is None:
        tol = 2*D.epsilon()
    tol = D.to_float(tol)
    xshape = D.shape(x0)
    xdim   = 1
    for __d in xshape:
        xdim *= __d
    x  = D.reshape(x0, (xdim, 1))
    if D.backend() == 'numpy' and "gdual_double" in D.available_float_fmt():
        is_vectorised = D.any(D.array([type(i[0]) == D.gdual_vdouble for i in x], dtype=D.bool))
        is_vectorised = is_vectorised or "vdouble" in D.float_fmt()
        is_gdual      = D.any(D.array([type(i[0]) in [D.gdual_double, D.gdual_vdouble, object] for i in x], dtype=D.bool))
        is_gdual      = is_gdual or "gdual" in D.float_fmt() or x.dtype == object
    else:
        is_vectorised = False
        is_gdual      = False
    
    fshape = D.shape(f(x0))
    fdim   = 1
    for __d in fshape:
        fdim *= __d
        
    def fun(x, __f=None):
        if D.backend() == 'torch' and not x.requires_grad:
            x.requires_grad = True
        return D.reshape(f(D.reshape(x, xshape)), (fdim, 1))
    
    if jac is None:
        if D.backend() == 'torch':
            def fun_jac(x, __f=None):
                if __f is None:
                    __f = fun(x)
                return D.reshape(D.jacobian(__f, x), (fdim, xdim))
        else:
            fun_jac = utilities.JacobianWrapper(fun, atol=tol, rtol=tol)
    else:
        jac_shape = D.shape(jac(x0))
        if is_gdual or jac_shape != (fdim, xdim):
            def fun_jac(x, __f=None):
                __j = D.reshape(jac(D.reshape(x, xshape)), (fdim, xdim))
                if is_gdual and not is_vectorised:
                    __j = D.to_float(__j)
                return __j
        else:
            fun_jac = lambda y, **kwargs: jac(y)
        
    if use_scipy and D.backend() == 'numpy' and not is_vectorised and not is_gdual:
        res = scipy.optimize.root(lambda __x: fun(__x)[:, 0], x[:,0], jac=fun_jac, tol=tol)
        init_iter = res.nfev + res.njev
        x = D.reshape(res.x, (xdim, 1))
        if res.success:
            return D.reshape(res.x, xshape), (res.success, init_iter, D.abs(res.qtf))
    else:
        init_iter = 0
    
    w_relax = 0.5
    F0      = fun(x)
    Jf0     = fun_jac(x, __f=F0)
    F1, Jf1 = D.copy(F0), D.copy(Jf0)
    Fn0     = D.to_float((F1.T@F1).item()**0.5)
    Fn1     = D.to_float(Fn0)
    dx      = D.zeros_like(x)
    dxn     = D.to_float((dx.T@dx).item()**0.5)
    I       = D.diag(D.ones_like(D.diag(Jf1)))
    iteration = 0
    fail_iter = 0
    
    for iteration in range(init_iter, maxiter):
        if verbose:
            df =  F1 -  F0
            dJ = Jf1 - Jf0
            df = (df.T@df).item()**0.5
            dJ = D.sum(dJ**2)**0.5
            print("[{iteration}]: x = {x}, f = {F1}, ||dx|| = {dxn}, ||F|| = {Fn1}, ||dF|| = {df}, ||dJ|| = {dJ}".format(**locals()))
            print()
        if 'vdouble' not in D.float_fmt() and estimate_cond(Jf1) > 100:
            P = preconditioner(Jf1)
        else:
            P = I
        sparse = (1.0 - D.sum(D.abs(D.to_float(Jf1)) > 0) / (xdim * fdim)) <= 0.7
        dx  = D.reshape(D.solve_linear_system(P@Jf1, -P@F1, sparse=sparse), (xdim, 1))
        x_best = x
        no_progress = True
        F0  = F1
        Fn0 = Fn1
        for __dx in [dx/m for m in [1.0,2.0,4.0,8.0]]:
            __x  = x + __dx
            __f  = fun(__x)
            __fn = D.to_float((__f.T@__f).item()**0.5)
            if __fn < Fn1:
                x   = __x
                dx  = __dx
                F1  = __f
                Fn1 = __fn
                no_progress = False
                break
        dxn = D.to_float((dx.T@dx).item()**0.5)
        if no_progress:
            fail_iter += 1
        if iteration % jac_update_rate == 0 or no_progress:
            Jf0, Jf1 = Jf0, fun_jac(x, __f=F1)
        else:
#             Jf0, Jf1 = D.copy(Jf1), Jf1 + ((F1 - F0) - Jf1@dx)@dx.T/dxn
            y_ex = Jf1@dx
            y_is = (F1 - F0)
            kI = (y_is - y_ex) / D.sum(y_ex.T@y_ex)
            Jf0, Jf1 = Jf1, D.reshape((1 + kI*Jf1*dx)*Jf1, (fdim, xdim))
        if Fn1 <= tol or dxn <= 32*D.max(D.abs(D.to_float(x) * tol)) or not D.all(D.isfinite(D.to_float(dx))) or fail_iter > 10:
            if verbose:
                print("[finished]: ||F|| = {Fn1}, ||dx|| = {dxn}, x = {x}, F = {F0}".format(**locals()))
            break
    return x, (Fn1 <= tol or dxn <= 32*D.max(D.abs(x * tol)), iteration, Fn1)


