import numpy
import warnings
import numpy as np
import scipy.optimize
import scipy.linalg
import scipy.sparse.linalg
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    pass
from desolver import backend as D
from desolver.utilities import utilities

__all__ = [
    'brentsroot',
    'brentsrootvec',
    'newtontrustregion',
    'nonlinear_roots',
    'transform_to_bounded_x',
    'transform_to_unbounded_x',
    'transform_to_bounded_fn',
    'transform_to_bounded_jac',
]


def transform_to_bounded_x(x, lower_bound, upper_bound):
    return D.ar_numpy.where(
        lower_bound < upper_bound,
        D.ar_numpy.arcsin(2 * (x - lower_bound) / (upper_bound - lower_bound) - 1),
        x
    )


def transform_to_bounded_dx(x, lower_bound, upper_bound):
    dbound = 2 / (upper_bound - lower_bound)
    dtrf = dbound * (x - lower_bound) - 1
    return D.ar_numpy.where(
        lower_bound < upper_bound,
        dbound * D.ar_numpy.reciprocal(D.ar_numpy.sqrt(1 - D.ar_numpy.square(dtrf))),
        1.0
    )


def transform_to_unbounded_x(x, lower_bound, upper_bound):
    return D.ar_numpy.where(
        lower_bound < upper_bound,
        0.5 * (D.ar_numpy.sin(x) + 1) * (upper_bound - lower_bound) + lower_bound,
        x
    )


def transform_to_unbounded_dx(x, lower_bound, upper_bound):
    return D.ar_numpy.where(
        lower_bound < upper_bound,
        0.5 * D.ar_numpy.cos(x) * (upper_bound - lower_bound),
        1.0
    )


def transform_to_bounded_fn(fn, lower_bound, upper_bound):
    def bounded_fn(bx, *args, **kwargs):
        x = transform_to_unbounded_x(bx, lower_bound, upper_bound)
        return fn(x, *args, **kwargs)
    return bounded_fn


def transform_to_bounded_jac(jac, lower_bound, upper_bound):
    def bounded_jac(bx, *args, **kwargs):
        x = transform_to_unbounded_x(bx, lower_bound, upper_bound)
        dx = transform_to_unbounded_dx(bx, lower_bound, upper_bound)
        return jac(x, *args, **kwargs) * D.ar_numpy.diag(dx[:,0])
    return bounded_jac


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
        tol = D.epsilon(lower_bound.dtype)
    if tol < D.epsilon(lower_bound.dtype):
        tol = D.epsilon(lower_bound.dtype)
    tol = D.ar_numpy.asarray(tol, like=lower_bound)
    a, b = D.ar_numpy.asarray(lower_bound), D.ar_numpy.asarray(upper_bound)
    fa = f(a)
    fb = f(b)

    if fa * fb >= D.epsilon(lower_bound.dtype):
        return D.ar_numpy.asarray(numpy.inf, like=lower_bound), False
    if D.ar_numpy.abs(fa) < D.ar_numpy.abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = D.ar_numpy.copy(a)
    d = D.ar_numpy.copy(b)
    fc = f(c)

    mflag = True
    conv = False
    numiter = 3

    while not conv:
        if verbose:
            with numpy.printoptions(precision=17, linewidth=200):
                print(f"[{numiter}] a={D.ar_numpy.to_numpy(a)}, b={D.ar_numpy.to_numpy(b)}, f(a)={D.ar_numpy.to_numpy(fa)}, f(b)={D.ar_numpy.to_numpy(fb)}")
        if fa != fc and fb != fc:
            s = (a * fb * fc) / ((fa - fb) * (fa - fc))
            s = s + (b * fa * fc) / ((fb - fa) * (fb - fc))
            s = s + (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = D.ar_numpy.abs(s - b) >= D.ar_numpy.abs(b - c) / 2
        cond3 = D.ar_numpy.abs(s - b) >= D.ar_numpy.abs(c - d) / 2
        cond4 = D.ar_numpy.abs(b - c) < tol
        cond5 = D.ar_numpy.abs(c - d) < tol
        bisect_now = cond1 or (mflag and cond2) or (not mflag and cond3) or (mflag and cond4) or (not mflag and cond5)
        mflag = bisect_now
        if mflag:
            s = (a + b) / 2

        fs = f(s)
        numiter += 1
        d = c

        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        if D.ar_numpy.abs(fa) < D.ar_numpy.abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        conv = (fb == 0 or fs == 0 or D.ar_numpy.abs(b - a) < tol)
        if numiter >= 64:
            break
    if verbose:
        with numpy.printoptions(precision=17, linewidth=200):
            print(f"[{numiter}] a={D.ar_numpy.to_numpy(a)}, b={D.ar_numpy.to_numpy(b)}, f(a)={D.ar_numpy.to_numpy(fa)}, f(b)={D.ar_numpy.to_numpy(fb)}")
    if return_interval:
        return b, D.ar_numpy.abs(f(b)) <= tol, (a, b)
    else:
        return b, D.ar_numpy.abs(f(b)) <= tol


def brentsrootvec(f, bounds, tol=None, verbose=False, return_interval=False, accepts_mask=False):
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
    accepts_mask : bool
        set to true to pass a `mask` array to the function, for certain situations,
        this avoids redundant computation for converged roots

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
        tol = D.epsilon(lower_bound.dtype)
    if tol < D.epsilon(lower_bound.dtype):
        tol = D.epsilon(lower_bound.dtype)
    tol = D.ar_numpy.asarray(tol, like=lower_bound)
    a, b = D.ar_numpy.asarray(lower_bound, like=tol), D.ar_numpy.asarray(upper_bound, like=tol)
    
    if isinstance(f, list):
        def _f(x, mask=None):
            out = [f[i](x[i]) if mask is None or (mask is not None and mask[i]) else D.ar_numpy.zeros_like(x[i]) for i in range(x.shape[0])]
            out = list(map(D.ar_numpy.atleast_1d, out))
            return D.ar_numpy.concatenate(out)
        if len(a.shape) == 0:
            a, b = a[None], b[None]
        if a.shape[0] == 1:
            a, b = D.ar_numpy.tile(a, (len(f),)), D.ar_numpy.tile(b, (len(f),))
    elif accepts_mask:
        _f = f
    else:
        def _f(x, mask=None):
            if mask is None:
                return f(x)
            return D.ar_numpy.where(mask, f(x), 0.0, like=x)

    if verbose:
        print(_f(a))

    conv = D.ar_numpy.ones_like(a, dtype=bool)

    fa = _f(a)
    fb = _f(b)

    mask = (D.ar_numpy.abs(fa) < D.ar_numpy.abs(fb))
    a[mask], b[mask] = b[mask], a[mask]
    fa[mask], fb[mask] = fb[mask], fa[mask]

    c = D.ar_numpy.copy(a)
    d = D.ar_numpy.copy(b)
    s = D.ar_numpy.copy(a)
    fc = _f(c)
    fs = D.ar_numpy.copy(fc)

    mflag = D.ar_numpy.ones_like(a, dtype=bool, like=upper_bound)
    conv[fa * fb >= 0] = False
    not_conv = D.ar_numpy.logical_not(conv)
    numiter = D.ar_numpy.ones_like(a, dtype=D.autoray.to_backend_dtype('int64', like=upper_bound), like=upper_bound) * 3
    true_conv = D.ar_numpy.abs(fb) <= tol

    while D.ar_numpy.any(conv):
        if verbose:
            with numpy.printoptions(precision=17, linewidth=200):
                print(f"[{numiter}] a={D.ar_numpy.to_numpy(a)}, b={D.ar_numpy.to_numpy(b)}, f(a)={D.ar_numpy.to_numpy(fa)}, f(b)={D.ar_numpy.to_numpy(fb)}, conv={D.ar_numpy.to_numpy(not_conv)}")
        mask = D.ar_numpy.logical_and(fa != fc, fb != fc)
        mask[not_conv] = False
        s[mask] = (a[mask] * fb[mask] * fc[mask]) / ((fa[mask] - fb[mask]) * (fa[mask] - fc[mask]))
        s[mask] = s[mask] + (b[mask] * fa[mask] * fc[mask]) / ((fb[mask] - fa[mask]) * (fb[mask] - fc[mask]))
        s[mask] = s[mask] + (c[mask] * fa[mask] * fb[mask]) / ((fc[mask] - fa[mask]) * (fc[mask] - fb[mask]))
        mask = D.ar_numpy.logical_not(mask)
        mask[D.ar_numpy.logical_not(conv)] = False
        s[mask] = b[mask] - fb[mask] * (b[mask] - a[mask]) / (fb[mask] - fa[mask])

        cond1 = D.ar_numpy.logical_not(
            D.ar_numpy.logical_or(D.ar_numpy.logical_and((3 * a + b) / 4 < s, s < b), D.ar_numpy.logical_and(b < s, s < (3 * a + b) / 4)))
        mask = cond1
        cond2 = D.ar_numpy.logical_and(mflag, D.ar_numpy.abs(s - b) >= D.ar_numpy.abs(b - c) / 2)
        mask = D.ar_numpy.logical_or(mask, cond2)
        cond3 = D.ar_numpy.logical_and(D.ar_numpy.logical_not(mflag), D.ar_numpy.abs(s - b) >= D.ar_numpy.abs(c - d) / 2)
        mask = D.ar_numpy.logical_or(mask, cond3)
        cond4 = D.ar_numpy.logical_and(mflag, D.ar_numpy.abs(b - c) < tol)
        mask = D.ar_numpy.logical_or(mask, cond4)
        cond5 = D.ar_numpy.logical_and(D.ar_numpy.logical_not(mflag), D.ar_numpy.abs(c - d) < tol)
        mask = D.ar_numpy.logical_or(mask, cond5)
        mask[not_conv] = False
        s[mask] = (a[mask] + b[mask]) / 2
        mflag[mask] = True
        mask = D.ar_numpy.logical_not(mask)
        mask[not_conv] = False
        mflag[mask] = False

        fs = _f(s, conv)
        numiter[conv] = numiter[conv] + 1
        d = c

        mask = fa * fs < 0
        mask[not_conv] = False
        b[mask] = s[mask]
        fb[mask] = fs[mask]
        mask = D.ar_numpy.logical_not(mask)
        mask[not_conv] = False
        a[mask] = s[mask]
        fa[mask] = fs[mask]

        mask = D.ar_numpy.abs(fa) < D.ar_numpy.abs(fb)
        mask[not_conv] = False
        a[mask], b[mask] = b[mask], a[mask]
        fa[mask], fb[mask] = fb[mask], fa[mask]

        conv = D.ar_numpy.logical_not(D.ar_numpy.logical_or(D.ar_numpy.logical_or(fb == 0, fs == 0), D.ar_numpy.abs(b - a) < tol))
        conv = conv & (numiter <= 64)
        not_conv = D.ar_numpy.logical_not(conv)
        true_conv = (D.ar_numpy.abs(fb) <= tol)

    if verbose:
        with numpy.printoptions(precision=17, linewidth=200):
            print(f"[{numiter}] a={D.ar_numpy.to_numpy(a)}, b={D.ar_numpy.to_numpy(b)}, f(a)={D.ar_numpy.to_numpy(fa)}, f(b)={D.ar_numpy.to_numpy(fb)}, conv={D.ar_numpy.to_numpy(not_conv)}")
    if return_interval:
        return b, true_conv, (a, b)
    else:
        return b, true_conv


# def preconditioner(A, tol=None):
#     if tol is None:
#         if D.epsilon() <= 1e-5:
#             tol = 32*D.epsilon()
#         else:
#             tol = D.epsilon()
#     if tol < 32*D.epsilon() and D.epsilon() <= 1e-5:
#         tol = 32*D.epsilon()
#     I    = D.eye(A.shape[0])
#     if D.backend() == 'torch':
#         I = I.to(A)
#     Pinv = D.ar_numpy.zeros_like(A)
#     A2   = A*A
#     nA   = 0.5 * (D.ar_numpy.sum(A2, axis=0)**0.5 + D.ar_numpy.sum(A2, axis=1)**0.5)
#     nA   = (nA > 32*D.epsilon())*nA + (nA <= 32*D.epsilon())
#     Pinv = D.ar_numpy.diag(nA)

#     Ik = Pinv@A
#     for _ in range(3):
#         AP = A@Pinv
#         Wn = -147*I + AP@(53*I + AP@(-11*I + AP))
#         In0 = 0.75*Pinv + 0.25*0.25*Pinv@(32*I + AP@(-113*I + AP@(231*I + AP@(-301*I + AP@(259*I + AP@Wn)))))
#         In1 = 2*Pinv - Ik@Pinv
#         if D.ar_numpy.linalg.norm(D.ar_numpy.to_numpy(I - In0@A)) < D.ar_numpy.linalg.norm(D.ar_numpy.to_numpy(I - In1@A)):
#             In = In0
#         else:
#             In = In1
#         if D.ar_numpy.linalg.norm(D.ar_numpy.to_numpy(I - In@A)) >= D.ar_numpy.linalg.norm(D.ar_numpy.to_numpy(I - Ik)):
#             break
#         else:
#             Pinv = In
#         nPinv = D.ar_numpy.linalg.norm(Pinv@A)
#         Pinv  = Pinv / nPinv
#         Ik    = Pinv@A
#         if D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(Ik))) - 1 <= tol:
#             break
#     return Pinv

# def estimate_cond(A):
#     out = D.ar_numpy.abs(A)
#     out = out[out > 0]
#     out = D.ar_numpy.sqrt(D.ar_numpy.max(out) / D.ar_numpy.min(out))
#     if out <= 32*D.epsilon():
#         out = D.ar_numpy.ones_like(out)
#     return out

def iterative_inverse_7th(A, Ainv0, maxiter=10):
    I = D.ar_numpy.diag(D.ar_numpy.ones_like(D.ar_numpy.diag(A)))
    Vn = Ainv0
    initial_norm = D.ar_numpy.linalg.norm(Vn @ A - I)
    for i in range(maxiter):
        Vn1 = (1 / 16) * Vn @ (120 * I + A @ Vn @ (-393 * I + A @ Vn @ (-861 * I + A @ Vn @ (
                    651 * I + A @ Vn @ (-315 * I + A @ Vn @ (931 * I + A @ Vn @ (-15 * I + A @ Vn)))))))
        new_norm = D.ar_numpy.linalg.norm(Vn1 @ A - I)
        if new_norm < D.tol_epsilon(A.dtype) or new_norm > initial_norm:
            break
        else:
            Vn = Vn1
            initial_norm = new_norm
    return Vn


# def iterative_inverse_1st(A, Ainv0, maxiter=10):
#     I = D.ar_numpy.diag(D.ar_numpy.ones_like(D.ar_numpy.diag(A)))
#     Vn = Ainv0
#     initial_norm = D.ar_numpy.linalg.norm(Vn @ A - I)
#     for i in range(maxiter):
#         Vn1 = Vn @ (2 * I - A @ Vn)
#         new_norm = D.ar_numpy.linalg.norm(Vn1 @ A - I)
#         if new_norm < D.tol_epsilon(A.dtype) or new_norm > initial_norm:
#             break
#         else:
#             Vn = Vn1
#             initial_norm = new_norm
#     return Vn


# def iterative_inverse_3rd(A, Ainv0, maxiter=10):
#     I = D.ar_numpy.diag(D.ar_numpy.ones_like(D.ar_numpy.diag(A)))
#     Vn = Ainv0
#     initial_norm = D.ar_numpy.linalg.norm(Vn @ A - I)
#     for i in range(maxiter):
#         Vn1 = Vn @ (2 * I - A @ Vn)
#         Vn1 = Vn1 @ (3 * I - A @ Vn1 @ (3 * I - A @ Vn1))
#         new_norm = D.ar_numpy.linalg.norm(Vn1 @ A - I)
#         if new_norm < D.tol_epsilon(A.dtype) or new_norm > initial_norm:
#             break
#         else:
#             Vn = Vn1
#             initial_norm = new_norm
#     return Vn


def broyden_update_jac(B, dx, df, Binv=None):
    y_ex = B @ dx
    y_is = df
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in matmul")
        kI = (y_is - y_ex) / D.ar_numpy.sum(y_ex.mT @ y_ex)
    B_new = D.ar_numpy.reshape((1 + kI * B * dx) * B, (df.shape[0], dx.shape[0]))
    if Binv is not None:
        Binv_new = Binv + ((dx - Binv @ y_is) / (y_is.mT @ y_is)) @ y_is.mT
        norm_val = D.ar_numpy.linalg.norm(Binv_new @ B_new - D.ar_numpy.diag(D.ar_numpy.ones_like(D.ar_numpy.diag(B))))
        if norm_val < 0.5:
            Binv_new = iterative_inverse_7th(B_new, Binv_new, maxiter=3)
        return B_new, Binv_new
    else:
        return B_new


def newtontrustregion(f, x0, jac=None, tol=None, verbose=False, maxiter=200, jac_update_rate=20, initial_trust_region=None, var_bounds=None):
    x0 = D.ar_numpy.asarray(x0)
    if tol is None:
        tol = D.tol_epsilon(x0.dtype)
    xshape = D.ar_numpy.shape(x0)
    if len(xshape) == 0:
        def f_vec(x):
            return D.ar_numpy.atleast_1d(f(x[0]))
        if jac is not None:
            def jac_vec(x):
                return D.ar_numpy.atleast_2d(jac(x[0]))
        else:
            jac_vec = None
        res = newtontrustregion(f_vec, D.ar_numpy.atleast_1d(x0), jac_vec, tol=tol, verbose=verbose, 
                                     maxiter=maxiter, initial_trust_region=initial_trust_region, var_bounds=var_bounds)
        return D.ar_numpy.reshape(res[0], xshape), res[1]
    xdim = 1
    for __d in xshape:
        xdim *= __d
    x = D.ar_numpy.reshape(x0, (xdim, 1))
    
    nfev = 0
    njev = 0

    fshape = D.ar_numpy.shape(f(x0))
    fdim = 1
    for __d in fshape:
        fdim *= __d
    
    inferred_backend = D.autoray.infer_backend(x0)

    def fun(x):
        nonlocal nfev
        nfev += 1
        return D.ar_numpy.reshape(f(D.ar_numpy.reshape(x, xshape)), (fdim, 1))

    if jac is None:
        if inferred_backend == 'torch':
            __fun_jac = torch.func.jacrev(fun, argnums=0)
        else:
            __fun_jac = utilities.JacobianWrapper(fun, atol=tol, rtol=tol, flat=True)
    else:
        __fun_jac = jac
    
    jac_shape = D.ar_numpy.shape(__fun_jac(x0))
    jacdim = 1
    for __d in jac_shape:
        jacdim *= __d
    if jacdim == fdim:
        def fun_jac(x):
            nonlocal njev
            __j = D.ar_numpy.reshape(__fun_jac(D.ar_numpy.reshape(x, xshape)), (fdim,))
            njev += 1
            return D.ar_numpy.diag(__j)
    elif jac_shape != (fdim, xdim):
        def fun_jac(x):
            nonlocal njev
            __j = D.ar_numpy.reshape(__fun_jac(D.ar_numpy.reshape(x, xshape)), (fdim, xdim))
            njev += 1
            return __j
    else:
        def fun_jac(x):
            nonlocal njev, jac
            njev += 1
            return __fun_jac(D.ar_numpy.reshape(x, xshape))
        
    if var_bounds is not None:
        var_bounds = [D.ar_numpy.asarray(var_bounds[0], like=x0), D.ar_numpy.asarray(var_bounds[1], like=x0)]
        if inferred_backend == 'torch':
            var_bounds = [
                var_bounds[0].to(x0.device, x0.dtype),
                var_bounds[1].to(x0.device, x0.dtype)
            ]
        fun = transform_to_bounded_fn(fun, *var_bounds)
        fun_jac = transform_to_bounded_jac(fun_jac, *var_bounds)
        x = transform_to_bounded_x(x, *var_bounds)

    F0 = fun(x)
    Jf0 = fun_jac(x)
    F1, Jf1 = D.ar_numpy.copy(F0), D.ar_numpy.copy(Jf0)
    Fn0 = D.ar_numpy.linalg.norm(F1).reshape(tuple())
    Fn1 = D.ar_numpy.copy(Fn0)
    dx = D.ar_numpy.zeros_like(x)
    dxn = D.ar_numpy.linalg.norm(dx).reshape(tuple())
    I = D.ar_numpy.diag(D.ar_numpy.ones_like(D.ar_numpy.diag(Jf1)))
    
    f64_type = D.autoray.to_backend_dtype('float64', like=inferred_backend)
    Jinv = D.ar_numpy.astype(D.ar_numpy.linalg.inv(D.ar_numpy.astype(Jf1, f64_type)), Jf1.dtype)
        
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")
        if D.ar_numpy.linalg.norm(Jinv @ Jf1 - I) < 0.5:
            Jinv = iterative_inverse_7th(Jf1, Jinv, maxiter=3)
    trust_region = 5.0 if initial_trust_region is None else initial_trust_region
    iteration = 0
    fail_iter = 0

    for iteration in range(0, maxiter):
        if verbose:
            df = F1 - F0
            dJ = Jf1 - Jf0
            df = (df.mT @ df).item() ** 0.5
            dJ = D.ar_numpy.sum(dJ ** 2) ** 0.5
            print(f"[ntr-{iteration}]: x = {D.ar_numpy.to_numpy(x)}, f = {D.ar_numpy.to_numpy(F1)}, ||dx|| = {D.ar_numpy.to_numpy(dxn)}, ||F|| = {D.ar_numpy.to_numpy(Fn1)}, ||dF|| = {D.ar_numpy.to_numpy(df)}, ||dJ|| = {D.ar_numpy.to_numpy(dJ)}")
        sparse = (1.0 - D.ar_numpy.sum(D.ar_numpy.abs(Jf1) > 0) / (xdim * fdim)) <= 0.7
        P = Jf1
        diagP = D.ar_numpy.diag(trust_region * D.ar_numpy.diag(P))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")
            warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)
            warnings.filterwarnings("ignore", category=scipy.sparse.linalg.MatrixRankWarning)
            dx = D.ar_numpy.reshape(D.ar_numpy.solve_linear_system(Jinv @ (P + diagP), -Jinv @ F1, sparse=sparse), (xdim, 1))
        no_progress = True
        F0 = F1
        Fn0 = Fn1
        for __dx in [dx / m for m in [1.0, 8.0, 64.0]]:
            __x = x + __dx
            __f = fun(__x)
            __fn = D.ar_numpy.linalg.norm(__f).reshape(tuple())
            if __fn < Fn1:
                x = __x
                dx = __dx
                F1 = __f
                Fn1 = __fn
                no_progress = False
                break
        dxn = D.ar_numpy.linalg.norm(dx).reshape(tuple())
        if no_progress:
            fail_iter += 1
        y_ex = Jf1 @ dx
        y_is = F1 - F0
        tr_ratio = D.ar_numpy.linalg.norm(y_ex).item()
        denom = D.ar_numpy.linalg.norm(y_is).item()
        failed_to_achieve_tol = denom <= (fdim + D.ar_numpy.linalg.norm(F0)) * tol
        if failed_to_achieve_tol:
            trust_region *= 0.9
        else:
            tr_ratio = tr_ratio / denom
            if tr_ratio > 0.5:
                trust_region *= 0.8 * 0.5 / tr_ratio
            elif tr_ratio < 0.25:
                trust_region *= 0.25 / tr_ratio
        if iteration % jac_update_rate == 0 or no_progress:
            Jf0, Jf1 = Jf0, fun_jac(x)
            Jinv = D.ar_numpy.astype(D.ar_numpy.linalg.inv(D.ar_numpy.astype(Jf1, f64_type)), Jf1.dtype)
        else:
            Jf0, (Jf1, Jinv) = Jf1, broyden_update_jac(Jf1, dx, F1 - F0, Jinv)
        xtol = tol * (xdim + D.ar_numpy.linalg.norm(x))
        success = dxn <= 0.8 * xtol
        success = success or Fn1 < 0.8 * tol
        convergence_failure = not D.ar_numpy.isfinite(dxn) or fail_iter > 2
        if success or convergence_failure:
            if verbose:
                print(f"[ntr-finished]: x = {D.ar_numpy.to_numpy(x)}, ||dx|| = {D.ar_numpy.to_numpy(dxn)}, ||F|| = {D.ar_numpy.to_numpy(Fn1)}, ||dF|| = {D.ar_numpy.to_numpy(df)}")
            break
    x = D.ar_numpy.reshape(x, xshape)
    if var_bounds is not None:
        x = transform_to_unbounded_x(x, *var_bounds)
    return x, (success and not convergence_failure, iteration, nfev, njev, Fn1)


def hybrj(f, x0, jac, tol=None, verbose=False, maxiter=200, var_bounds=None):
    x0 = D.ar_numpy.asarray(x0)
    if tol is None:
        tol = D.tol_epsilon(x0.dtype)
    xshape = D.ar_numpy.shape(x0)
    xdim = 1
    for __d in xshape:
        xdim *= __d
    x = D.ar_numpy.reshape(x0, (xdim, 1))
    
    fshape = D.ar_numpy.shape(f(x0))
    fdim = 1
    for __d in fshape:
        fdim *= __d
    
    inferred_backend = D.autoray.infer_backend(x0)

    def fun(x):
        return D.ar_numpy.reshape(f(D.ar_numpy.reshape(x, xshape)), (fdim, 1))

    if jac is None:
        if inferred_backend == 'torch':
            __fun_jac = torch.func.jacrev(fun, argnums=0)
        else:
            __fun_jac = utilities.JacobianWrapper(fun, atol=tol, rtol=tol, flat=True)
    else:
        __fun_jac = jac
    
    jac_shape = D.ar_numpy.shape(__fun_jac(x0))
    jacdim = 1
    for __d in jac_shape:
        jacdim *= __d
    if jacdim == fdim:
        def fun_jac(x):
            __j = D.ar_numpy.reshape(__fun_jac(D.ar_numpy.reshape(x, xshape)), (fdim,))
            return D.ar_numpy.diag(__j)
    elif jac_shape != (fdim, xdim):
        def fun_jac(x):
            __j = D.ar_numpy.reshape(__fun_jac(D.ar_numpy.reshape(x, xshape)), (fdim, xdim))
            return __j
    else:
        def fun_jac(x):
            nonlocal jac
            return __fun_jac(D.ar_numpy.reshape(x, xshape))
        
    if var_bounds is not None:
        var_bounds = [D.ar_numpy.asarray(var_bounds[0], like=x0), D.ar_numpy.asarray(var_bounds[1], like=x0)]
        if inferred_backend == 'torch':
            var_bounds = [
                var_bounds[0].to(x0.device, x0.dtype),
                var_bounds[1].to(x0.device, x0.dtype)
            ]
        fun = transform_to_bounded_fn(fun, *var_bounds)
        fun_jac = transform_to_bounded_jac(fun_jac, *var_bounds)
        x = transform_to_bounded_x(x, *var_bounds)
    
    F0 = fun(x)
    F1 = D.ar_numpy.copy(F0)
    J0 = fun_jac(x)
    dx = D.ar_numpy.zeros_like(x)
    dxn = D.ar_numpy.linalg.norm(dx)

    trust_region = D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.diag(J0)))
    iteration = 0
    success = False
    for iteration in range(maxiter):
        if verbose:
            df = D.ar_numpy.linalg.norm(F1 - F0)
            Fn0 = D.ar_numpy.linalg.norm(F0)
            print(f"[hybrj-{iteration}]: tr = {D.ar_numpy.to_numpy(trust_region)}, x = {D.ar_numpy.to_numpy(x)}, f = {D.ar_numpy.to_numpy(F1)}, ||dx|| = {D.ar_numpy.to_numpy(dxn)}, ||F|| = {D.ar_numpy.to_numpy(Fn0)}, ||dF|| = {D.ar_numpy.to_numpy(df)}")
        Jt_mul_F = J0.mT @ F0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in matmul")
            warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)
            warnings.filterwarnings("ignore", category=scipy.sparse.linalg.MatrixRankWarning)
            dx_gn = -D.ar_numpy.solve_linear_system(J0.mT @ J0, Jt_mul_F)
        dx_sd = -Jt_mul_F
        tparam = -dx_sd.mT @ Jt_mul_F / D.ar_numpy.linalg.norm(J0 @ dx_sd) ** 2
        xtol = tol * (xdim + D.ar_numpy.linalg.norm(x))
        if D.ar_numpy.all(D.ar_numpy.linalg.norm(dx_gn) <= trust_region) or D.ar_numpy.linalg.norm(dx_gn - tparam * dx_sd) < xtol:
            dx = dx_gn
        elif D.ar_numpy.all(D.ar_numpy.linalg.norm(dx_sd) >= trust_region):
            dx = trust_region * dx_sd / D.ar_numpy.linalg.norm(dx_sd)
        else:
            a = tparam * dx_sd
            an = D.ar_numpy.minimum(D.ar_numpy.linalg.norm(a) ** 2, trust_region)
            b = dx_gn - a
            bn = D.ar_numpy.linalg.norm(b) ** 2
            c = (a.mT @ b)
            if c <= 0:
                s = (-c + D.ar_numpy.sqrt(c ** 2 + bn * (trust_region - an))) / bn
            else:
                s = (trust_region - an) / (c + D.ar_numpy.sqrt(c ** 2 + bn * (trust_region - an)))
            dx = tparam * dx_sd + s * (dx_gn - tparam * dx_sd)
        __x = x + dx.reshape(xdim, 1)
        __f = fun(__x)
        dxn = D.ar_numpy.linalg.norm(dx)
        y_ex = J0 @ dx
        y_is = __f - F0
        gain = 2.0 * D.ar_numpy.linalg.norm(y_is) / (D.ar_numpy.linalg.norm(F0) - D.ar_numpy.linalg.norm(F0 + y_ex))
        no_progress = not (D.ar_numpy.max(gain) > 0)
        if not no_progress:
            x = __x
            F1 = F0
            F0 = __f
            success = D.ar_numpy.linalg.norm(F0) < tol or dxn <= xtol
        if no_progress:
            J0 = fun_jac(x)
        else:
            J0 = broyden_update_jac(J0, dx, y_is)
        if D.ar_numpy.any(~D.ar_numpy.isfinite(dx)):
            raise ValueError("Encountered nan!")
        if D.ar_numpy.max(gain) > 0.75:
            trust_region = D.ar_numpy.maximum(trust_region, 3 *  D.ar_numpy.linalg.norm(dx_gn))
        elif D.ar_numpy.max(gain) < 0.25:
            trust_region = trust_region * 0.5
            success = success or trust_region <= xtol
        if success:
            if verbose:
                Fn0 = D.ar_numpy.linalg.norm(F0)
                print(f"[hybrj-finished]: ||F|| = {D.ar_numpy.to_numpy(Fn0)}, ||dx|| = {D.ar_numpy.to_numpy(dxn)}, x = {D.ar_numpy.to_numpy(x)}, F = {D.ar_numpy.to_numpy(F0)}")
            break
    x = D.ar_numpy.reshape(x, xshape)
    if var_bounds is not None:
        x = transform_to_unbounded_x(x, *var_bounds)
    return x, (success, dxn, iteration, D.ar_numpy.reshape(F0, fshape))


def nonlinear_roots(f, x0, jac=None, tol=None, verbose=False, maxiter=200, use_scipy=True,
                    additional_args=tuple(), additional_kwargs=dict(), var_bounds=None):
    x0 = D.ar_numpy.asarray(x0)    
    if tol is None:
        tol = D.tol_epsilon(x0.dtype)
    xshape = D.ar_numpy.shape(x0)
    if len(xshape) == 0:
        def f_vec(x):
            return D.ar_numpy.atleast_1d(f(x[0]))
        if jac is not None:
            def jac_vec(x):
                return D.ar_numpy.atleast_2d(jac(x[0]))
        else:
            jac_vec = None
        res = nonlinear_roots(f_vec, D.ar_numpy.atleast_1d(x0), jac_vec, tol=tol, verbose=verbose, maxiter=maxiter)
        return D.ar_numpy.reshape(res[0], xshape), res[1]
    xdim = 1
    for __d in xshape:
        xdim *= __d
    x = D.ar_numpy.reshape(x0, (xdim, 1))

    __f0 = f(x0, *additional_args, **additional_kwargs)
    fshape = D.ar_numpy.shape(__f0)
    fdim = 1
    for __d in fshape:
        fdim *= __d
    nfev = 1
    njev = 0
        
    inferred_backend = D.autoray.infer_backend(x0)

    def fun(x):
        nonlocal nfev
        nfev += 1
        return D.ar_numpy.reshape(f(D.ar_numpy.reshape(x, xshape), *additional_args, **additional_kwargs), (fdim, 1))

    if jac is None:
        if inferred_backend == 'torch':
            __fun_jac = torch.func.jacrev(fun, argnums=0)
        else:
            __fun_jac = utilities.JacobianWrapper(fun, atol=tol, rtol=tol, flat=True)
    else:
        def __fun_jac(x):
            return jac(x, *additional_args, **additional_kwargs)
    
    jac_shape = D.ar_numpy.shape(__fun_jac(x0))
    jacdim = 1
    for __d in jac_shape:
        jacdim *= __d
    if jacdim == fdim:
        def fun_jac(x):
            nonlocal njev
            __j = D.ar_numpy.reshape(__fun_jac(D.ar_numpy.reshape(x, xshape)), (fdim,))
            njev += 1
            return D.ar_numpy.diag(__j)
    elif jac_shape != (fdim, xdim):
        def fun_jac(x):
            nonlocal njev
            __j = D.ar_numpy.reshape(__fun_jac(D.ar_numpy.reshape(x, xshape)), (fdim, xdim))
            njev += 1
            return __j
    else:
        def fun_jac(x):
            nonlocal njev, jac
            njev += 1
            return __fun_jac(D.ar_numpy.reshape(x, xshape))
        
    if var_bounds is not None:
        var_bounds = [D.ar_numpy.asarray(var_bounds[0], like=x0), D.ar_numpy.asarray(var_bounds[1], like=x0)]
        if inferred_backend == 'torch':
            var_bounds = [
                var_bounds[0].to(x0.device, x0.dtype),
                var_bounds[1].to(x0.device, x0.dtype)
            ]
        fun = transform_to_bounded_fn(fun, *var_bounds)
        fun_jac = transform_to_bounded_jac(fun_jac, *var_bounds)
        x = transform_to_bounded_x(x, *var_bounds)

    # Check if type is 128-bit float which is unsupported by scipy.minpack
    if use_scipy and inferred_backend == 'numpy':
        is_too_wide = np.finfo(x0.dtype).bits > 64
    else:
        is_too_wide = False
    
    if use_scipy and inferred_backend == 'numpy' and not is_too_wide:
        res = scipy.optimize.root(lambda __x: fun(__x[...,None])[...,0], x[...,0], jac=fun_jac, tol=tol)
        nfev = res.nfev
        njev = res.njev
        init_iter = res.nfev + res.njev
        x = D.ar_numpy.reshape(res.x, (xdim, 1))
        F = D.ar_numpy.reshape(res.fun, fshape)
        success = res.success or ("no futher improvement" in res.message and D.ar_numpy.linalg.norm(res.fun) <= D.tol_epsilon(x0.dtype))
        if success:
            x = D.ar_numpy.reshape(x, xshape)
            if var_bounds is not None:
                x = transform_to_unbounded_x(x, *var_bounds)
            return x, (success, init_iter, nfev, njev, D.ar_numpy.linalg.norm(F))
        else:
            x = D.ar_numpy.reshape(x0, (xdim, 1))
    else:
        root, (success, prec, iterations, F) = hybrj(fun, x, fun_jac, tol=tol, verbose=verbose, maxiter=maxiter)
        success = success or D.ar_numpy.linalg.norm(F) <= D.tol_epsilon(x0.dtype)
        if success:
            x = D.ar_numpy.reshape(root, xshape)
            if var_bounds is not None:
                x = transform_to_unbounded_x(x, *var_bounds)
            return x, (success, iterations, nfev, njev, prec)
        else:
            x = D.ar_numpy.reshape(x0, (xdim, 1))
    
    root, (success, iterations, *_, prec) = newtontrustregion(fun, x, jac=fun_jac, tol=tol, verbose=verbose, maxiter=maxiter, jac_update_rate=10, initial_trust_region=0.0)
    success = success or prec <= D.tol_epsilon(x0.dtype)
    
    x = D.ar_numpy.reshape(root, xshape)
    if var_bounds is not None:
        x = transform_to_unbounded_x(x, *var_bounds)
    
    return x, (success, iterations, nfev, njev, prec)
