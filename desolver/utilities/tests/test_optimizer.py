import desolver as de
import desolver.backend as D
import numpy as np
from desolver.utilities.tests import common
import pytest


def set_torch_printoptions():
    import torch
    torch.set_printoptions(precision=17)
    torch.autograd.set_detect_anomaly(True)


def convert_tolerance(tolerance, dtype):
    if tolerance is not None:
        tolerance = tolerance * D.tol_epsilon(dtype)
        tol = 32 * tolerance
    else:
        tol = 32 * D.tol_epsilon(dtype)
    return tolerance, tol


@pytest.mark.slow
@common.test_fn_param
def test_rootfinding_transforms(fn, dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()
    
    x0 = D.ar_numpy.asarray(fn.root_interval[0] + 0.5 * (fn.root_interval[1] - fn.root_interval[0]), dtype=dtype_var, like=backend_var)[None,None]
    
    var_bounds = [
        D.ar_numpy.asarray(fn.root_interval[0], dtype=dtype_var, like=x0)[None,None],
        D.ar_numpy.asarray(fn.root_interval[1], dtype=dtype_var, like=x0)[None,None]
    ]
    if backend_var == 'torch':
        var_bounds = [
            var_bounds[0].to(x0.dtype),
            var_bounds[1].to(x0.dtype)
        ]
    
    tol = D.tol_epsilon(dtype_var)
    
    bx0 = de.utilities.optimizer.transform_to_bounded_x(x0, *var_bounds)
    blb = de.utilities.optimizer.transform_to_bounded_x(var_bounds[0], *var_bounds)
    bub = de.utilities.optimizer.transform_to_bounded_x(var_bounds[1], *var_bounds)
    
    neg_half_pi = D.ar_numpy.arcsin(-D.ar_numpy.ones_like(blb))
    pos_half_pi = D.ar_numpy.arcsin( D.ar_numpy.ones_like(bub))
    
    assert (D.ar_numpy.allclose(bx0, D.ar_numpy.zeros_like(bx0), tol, tol))
    assert (D.ar_numpy.allclose(blb, neg_half_pi,  tol, tol))
    assert (D.ar_numpy.allclose(bub, pos_half_pi,  tol, tol))
    assert (D.ar_numpy.allclose(x0, de.utilities.optimizer.transform_to_unbounded_x(bx0, *var_bounds), tol, tol))
    
    bfn = de.utilities.optimizer.transform_to_bounded_fn(fn, *var_bounds)
    assert (D.ar_numpy.allclose(fn(x0), bfn(bx0), tol, tol))
    assert (D.ar_numpy.allclose(fn(var_bounds[0]), bfn(blb), tol, tol))
    assert (D.ar_numpy.allclose(fn(var_bounds[1]), bfn(bub), tol, tol))
    
    bfn_jac = de.utilities.optimizer.transform_to_bounded_jac(fn.jac, *var_bounds)
    if backend_var == 'torch':
        import torch
        bfn_jac_wrapped = torch.func.jacrev(bfn, argnums=0)
    else:
        bfn_jac_wrapped = de.utilities.JacobianWrapper(bfn, atol=tol, rtol=tol, flat=True)
    
    assert (D.ar_numpy.allclose(bfn_jac_wrapped(bx0), bfn_jac(bx0), tol**0.5, tol**0.5))
    assert (D.ar_numpy.allclose(bfn_jac_wrapped(blb), bfn_jac(blb), tol**0.5, tol**0.5))
    assert (D.ar_numpy.allclose(bfn_jac_wrapped(bub), bfn_jac(bub), tol**0.5, tol**0.5))


def test_brentsroot_same_sign(dtype_var, backend_var, device_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()

    ac_prod = D.ar_numpy.asarray(np.random.uniform(0.9, 1.1), dtype=dtype_var, like=backend_var)
    a = D.ar_numpy.asarray(1.0, dtype=dtype_var, like=backend_var)
    b = D.ar_numpy.asarray(1.0, dtype=dtype_var, like=backend_var)
    
    if backend_var == 'torch':
        ac_prod = ac_prod.to(device_var)
        a = a.to(device_var)
        b = b.to(device_var)

    gt_root = -b / a
    lb, ub = -b / a - 1, -b / a - 2

    fun = lambda x: a * x + b

    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun(gt_root)))) <= D.tol_epsilon(dtype_var)

    root, success = de.utilities.optimizer.brentsroot(fun, [lb, ub], D.tol_epsilon(dtype_var), verbose=0)

    assert (np.isinf(D.ar_numpy.to_numpy(root)))
    assert (not success)


def test_brentsroot_epsilon_too_small(dtype_var, backend_var, device_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()

    ac_prod = D.ar_numpy.asarray(np.random.uniform(0.9, 1.1), dtype=dtype_var, like=backend_var)
    a = D.ar_numpy.asarray(1.0, dtype=dtype_var, like=backend_var)
    b = D.ar_numpy.asarray(1.0, dtype=dtype_var, like=backend_var)
    
    if backend_var == 'torch':
        ac_prod = ac_prod.to(device_var)
        a = a.to(device_var)
        b = b.to(device_var)

    gt_root = -b / a
    lb, ub = -b / a - 1, -b / a - 2

    fun = lambda x: a * x + b

    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun(gt_root)))) <= D.tol_epsilon(dtype_var)

    root, success = de.utilities.optimizer.brentsroot(fun, [lb, ub], D.tol_epsilon(dtype_var)*1e-7, verbose=0)

    assert (np.isinf(D.ar_numpy.to_numpy(root)))
    assert (not success)


def test_brentsroot_wrong_order(dtype_var, backend_var, device_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()
        
    a = D.ar_numpy.asarray(1.0, dtype=dtype_var, like=backend_var)
    b = D.ar_numpy.asarray(1.0, dtype=dtype_var, like=backend_var)
    
    if backend_var == 'torch':
        a = a.to(device_var)
        b = b.to(device_var)

    gt_root = -b / a
    lb, ub = -b / a - 1, -b / a + 1

    fun = lambda x: a * x + b

    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun(gt_root)))) <= D.tol_epsilon(dtype_var)

    root, success = de.utilities.optimizer.brentsroot(fun, [ub, lb], D.tol_epsilon(dtype_var), verbose=0)

    assert (success)
    assert (np.allclose(D.ar_numpy.to_numpy(gt_root), D.ar_numpy.to_numpy(root), D.tol_epsilon(dtype_var), D.tol_epsilon(dtype_var)))


@pytest.mark.parametrize('tolerance', [None, 100.0, 10.0, 1.0])
@pytest.mark.parametrize('ac_prod_val', np.linspace(0.9, 1.1, 4))
@pytest.mark.parametrize('a_val', [-1.0, 1.0])
def test_brentsroot(tolerance, dtype_var, backend_var, device_var, a_val, ac_prod_val):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()
    
    tolerance, tol = convert_tolerance(tolerance, dtype_var)

    ac_prod = D.ar_numpy.asarray(ac_prod_val, dtype=dtype_var, like=backend_var)
    a = D.ar_numpy.asarray(a_val, dtype=dtype_var, like=backend_var)
    if backend_var == 'torch':
        ac_prod = ac_prod.to(device_var)
        a = a.to(device_var)
    c = ac_prod / a
    b = D.ar_numpy.sqrt(0.01 + 4 * ac_prod)

    gt_root = -b / (2 * a) - 0.1 / (2 * a)

    ub = -b / (2 * a)
    lb = -b / (2 * a) - 1.0 / (2 * a)

    fun = lambda x: a * x ** 2 + b * x + c

    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun(gt_root))) <= D.tol_epsilon(dtype_var))

    root, success = de.utilities.optimizer.brentsroot(fun, [lb, ub], tolerance, verbose=1)

    assert (success)
    assert (np.allclose(D.ar_numpy.to_numpy(gt_root), D.ar_numpy.to_numpy(root), tol, tol))
    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun(root))) <= tol)

    # Check interval
    root, success, (root_lb, root_ub) = de.utilities.optimizer.brentsroot(fun, [lb, ub], tolerance, verbose=0, return_interval=True)

    assert (success)
    assert (D.ar_numpy.all((root_lb - 32*tol <= gt_root) & (gt_root <= root_ub + 32*tol)))
    assert (np.allclose(D.ar_numpy.to_numpy(gt_root), D.ar_numpy.to_numpy(root), tol, tol))
    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun(root))) <= tol)


@pytest.mark.parametrize('tolerance', [None, 100.0, 10.0, 1.0])
def test_brentsrootvec(tolerance, dtype_var, backend_var, device_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()
    
    tolerance, tol = convert_tolerance(tolerance, dtype_var)
    
    ac_prod_vals = np.linspace(0.9, 1.1, 8)
    a_vals = [-1, 1]
    
    ac_prod_vals, a_vals = [
        D.ar_numpy.asarray([v for v in ac_prod_vals for _ in a_vals], dtype=dtype_var, like=backend_var),
        D.ar_numpy.asarray([v for _ in ac_prod_vals for v in a_vals], dtype=dtype_var, like=backend_var)
    ]
    
    if backend_var == 'torch':
        ac_prod_vals = ac_prod_vals.to(device_var)
        a_vals = a_vals.to(device_var)
    
    c_vals = ac_prod_vals / a_vals
    b_vals = D.ar_numpy.sqrt(0.01 + 4 * ac_prod_vals)

    gt_root = -b_vals / (2 * a_vals) - 0.1 / (2 * a_vals)

    ub = -b_vals / (2 * a_vals)
    lb = -b_vals / (2 * a_vals) - 1.0 / (2 * a_vals)

    # As list of functions
    fun_list = [(lambda a, b, c: lambda x: a * x ** 2 + b * x + c)(a, b, c) for a, b, c in zip(a_vals, b_vals, c_vals)]

    assert (all(map((lambda i: D.ar_numpy.to_numpy(D.ar_numpy.abs(i)) <= D.tol_epsilon(dtype_var)),
                map((lambda x: x[0](x[1])), zip(fun_list, gt_root)))))

    root_list, success = de.utilities.optimizer.brentsrootvec(fun_list,
                                                                [lb, ub],
                                                                tolerance, verbose=1)

    assert (np.all(D.ar_numpy.to_numpy(success)))
    assert (np.allclose(D.ar_numpy.to_numpy(gt_root), D.ar_numpy.to_numpy(root_list), tol, tol))

    assert (all(map((lambda i: D.ar_numpy.to_numpy(D.ar_numpy.abs(i)) <= tol),
                    map((lambda x: x[0](x[1])), zip(fun_list, root_list)))))
    
    
    fun_vec = lambda x: a_vals * x**2 + b_vals * x + c_vals

    assert np.allclose(D.ar_numpy.to_numpy(fun_vec(gt_root)), 0.0, D.tol_epsilon(dtype_var), D.tol_epsilon(dtype_var))

    root_list, success = de.utilities.optimizer.brentsrootvec(fun_vec,
                                                                [lb, ub],
                                                                tolerance, verbose=0)
    assert (np.all(D.ar_numpy.to_numpy(success)))
    assert (np.allclose(D.ar_numpy.to_numpy(gt_root), D.ar_numpy.to_numpy(root_list), tol, tol))

    assert np.allclose(D.ar_numpy.to_numpy(fun_vec(root_list)), 0.0, tol, tol)

    if tolerance is not None:
        root_list, success = de.utilities.optimizer.brentsrootvec(fun_vec,
                                                                    [lb, ub],
                                                                    tolerance * 1e-7, verbose=0)

        assert (np.all(D.ar_numpy.to_numpy(success)))
        assert (np.allclose(D.ar_numpy.to_numpy(gt_root), D.ar_numpy.to_numpy(root_list), tol, tol))

        assert np.allclose(D.ar_numpy.to_numpy(fun_vec(root_list)), 0.0, tol, tol)

    root_list, success, (root_lb, root_ub) = de.utilities.optimizer.brentsrootvec(fun_vec,
                                                                [lb, ub],
                                                                tolerance, verbose=0, return_interval=True)

    assert (np.all(D.ar_numpy.to_numpy(success)))
    assert (D.ar_numpy.all((root_lb - 32*tol <= gt_root) & (gt_root <= root_ub + 32*tol)))
    assert (np.allclose(D.ar_numpy.to_numpy(gt_root), D.ar_numpy.to_numpy(root_list), tol, tol))
    assert np.allclose(D.ar_numpy.to_numpy(fun_vec(root_list)), 0.0, tol, tol)


@pytest.mark.slow
@pytest.mark.parametrize('tolerance', [None, 4.0])
@pytest.mark.parametrize('ac_prod_val', np.linspace(0.9, 1.1, 4))
@pytest.mark.parametrize('a_val', [-1.0, 1.0])
@pytest.mark.parametrize('solver', [de.utilities.optimizer.nonlinear_roots, de.utilities.optimizer.newtontrustregion, de.utilities.optimizer.hybrj])
def test_nonlinear_root(solver, tolerance, dtype_var, backend_var, device_var, a_val, ac_prod_val):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()
    
    tolerance, tol = convert_tolerance(tolerance, dtype_var)

    ac_prod = D.ar_numpy.asarray(ac_prod_val, dtype=dtype_var, like=backend_var)
    a = D.ar_numpy.asarray(a_val, dtype=dtype_var, like=backend_var)
    if backend_var == 'torch':
        ac_prod = ac_prod.to(device_var)
        a = a.to(device_var)
    c = ac_prod / a
    b = D.ar_numpy.sqrt(0.01 + 4 * ac_prod)

    gt_root1 = -b / (2 * a) - 0.1 / (2 * a)
    gt_root2 = -b / (2 * a) + 0.1 / (2 * a)
    
    ub = -b / (2 * a) - 0.2 / (2 * a)
    lb = -b / (2 * a) - 0.4 / (2 * a)

    fun_fn = lambda x: a * x ** 2 + b * x + c
    jac_fn = lambda x: 2 * a * x + b

    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(gt_root1))) <= D.tol_epsilon(dtype_var))
    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(gt_root2))) <= D.tol_epsilon(dtype_var))

    for test_idx in range(4):
        x0 = lb + test_idx * (ub - lb) / 4
        if backend_var == 'torch':
            x0 = x0.to(device_var)

        root, (success, *_) = solver(fun_fn, x0, jac=jac_fn, tol=tolerance, verbose=1)

        assert (success)
        
        conv_root1 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root1), tol, tol)
        conv_root2 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root2), tol, tol)
        assert (conv_root1 or conv_root2)
        assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(root))) <= tol)


@pytest.mark.slow
@pytest.mark.parametrize('tolerance', [100.0, 1e4])
@pytest.mark.parametrize('ac_prod_val', np.linspace(0.9, 1.1, 3))
@pytest.mark.parametrize('a_val', [-1.0, 1.0])
@pytest.mark.parametrize('solver', [de.utilities.optimizer.newtontrustregion, de.utilities.optimizer.hybrj, de.utilities.optimizer.nonlinear_roots])
@pytest.mark.parametrize('shape', [(1,), (4,4), (2,3,5)])
def test_nonlinear_root_dims(solver, tolerance, dtype_var, backend_var, device_var, a_val, ac_prod_val, shape):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()
    
    tolerance, tol = convert_tolerance(tolerance, dtype_var)

    ac_prod = D.ar_numpy.tile(D.ar_numpy.asarray(ac_prod_val, dtype=dtype_var, like=backend_var), shape)
    a = D.ar_numpy.tile(D.ar_numpy.asarray(a_val, dtype=dtype_var, like=backend_var), shape)
    if backend_var == 'torch':
        ac_prod = ac_prod.to(device_var)
        a = a.to(device_var)
    c = ac_prod / a
    b = D.ar_numpy.sqrt(0.01 + 4 * ac_prod)

    gt_root1 = -b / (2 * a) - 0.1 / (2 * a)
    gt_root2 = -b / (2 * a) + 0.1 / (2 * a)
    
    ub = -b / (2 * a) - 0.2 / (2 * a)
    lb = -b / (2 * a) - 0.4 / (2 * a)

    fun_fn = lambda x: a * x ** 2 + b * x + c
    jac_fn = lambda x: 2 * a * x + b

    assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(gt_root1))) <= D.tol_epsilon(dtype_var))
    assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(gt_root2))) <= D.tol_epsilon(dtype_var))

    for test_idx in range(4):
        x0 = lb + test_idx * (ub - lb) / 4
        if backend_var == 'torch':
            x0 = x0.to(device_var)

        root, (success, *_) = solver(fun_fn, x0, jac=jac_fn, tol=tolerance, verbose=1)

        assert (success)
        
        conv_root1 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root1), tol, tol)
        conv_root2 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root2), tol, tol)
        
        assert D.ar_numpy.all(conv_root1 | conv_root2)
        assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(root))) <= tol)

        # Check with jacobian reshaped to be "strange"
        root, (success, *_) = solver(fun_fn, x0, jac=lambda *args, **kwargs: jac_fn(*args, **kwargs)[None,None], tol=tolerance, verbose=1)

        assert (success)
        
        conv_root1 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root1), tol, tol)
        conv_root2 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root2), tol, tol)
        
        assert D.ar_numpy.all(conv_root1 | conv_root2)
        assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(root))) <= tol)


@pytest.mark.slow
@pytest.mark.parametrize('tolerance', [None, 100.0, 4.0])
@pytest.mark.parametrize('ac_prod_val', np.linspace(0.9, 1.1, 3))
@pytest.mark.parametrize('a_val', [-1.0, 1.0])
@pytest.mark.parametrize('solver', [de.utilities.optimizer.nonlinear_roots, de.utilities.optimizer.newtontrustregion, de.utilities.optimizer.hybrj])
@pytest.mark.parametrize('shape', [(2,), (3,3),])
def test_nonlinear_root_dims_no_jacobian(solver, tolerance, dtype_var, backend_var, device_var, a_val, ac_prod_val, shape, requires_autodiff):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()
    
    tolerance, tol = convert_tolerance(tolerance, dtype_var)

    ac_prod = D.ar_numpy.tile(D.ar_numpy.asarray(ac_prod_val, dtype=dtype_var, like=backend_var), shape)
    a = D.ar_numpy.tile(D.ar_numpy.asarray(a_val, dtype=dtype_var, like=backend_var), shape)
    if backend_var == 'torch':
        ac_prod = ac_prod.to(device_var)
        a = a.to(device_var)
    c = ac_prod / a
    b = D.ar_numpy.sqrt(0.01 + 4 * ac_prod)

    gt_root1 = -b / (2 * a) - 0.1 / (2 * a)
    gt_root2 = -b / (2 * a) + 0.1 / (2 * a)
    
    ub = -b / (2 * a) - 0.2 / (2 * a)
    lb = -b / (2 * a) - 0.4 / (2 * a)

    fun_fn = lambda x: a * x ** 2 + b * x + c
    jac_fn = None  # lambda x: 2 * a * x + b

    assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(gt_root1))) <= D.tol_epsilon(dtype_var))
    assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(gt_root2))) <= D.tol_epsilon(dtype_var))

    for test_idx in range(4):
        x0 = lb + test_idx * (ub - lb) / 4
        if backend_var == 'torch':
            x0 = x0.to(device_var)

        root, (success, *_) = solver(fun_fn, x0, jac=jac_fn, tol=tolerance, verbose=1)

        assert (success)
        
        conv_root1 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root1), tol, tol)
        conv_root2 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root2), tol, tol)
        
        assert D.ar_numpy.all(conv_root1 | conv_root2)
        assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(root))) <= tol)


@pytest.mark.slow
@pytest.mark.parametrize('tolerance', [None, 100.0, 4.0])
@pytest.mark.parametrize('ac_prod_val', np.linspace(0.9, 1.1, 3))
@pytest.mark.parametrize('a_val', [-1.0, 1.0])
@pytest.mark.parametrize('solver', [de.utilities.optimizer.nonlinear_roots, de.utilities.optimizer.newtontrustregion, de.utilities.optimizer.hybrj])
def test_nonlinear_root_dims_no_jacobian_numpy(solver, tolerance, dtype_var, a_val, ac_prod_val):
    shape = (2,)
    backend_var = 'numpy'
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    
    tolerance, tol = convert_tolerance(tolerance, dtype_var)

    ac_prod = D.ar_numpy.tile(D.ar_numpy.asarray(ac_prod_val, dtype=dtype_var, like=backend_var), shape)
    a = D.ar_numpy.tile(D.ar_numpy.asarray(a_val, dtype=dtype_var, like=backend_var), shape)
    c = ac_prod / a
    b = D.ar_numpy.sqrt(0.01 + 4 * ac_prod)

    gt_root1 = -b / (2 * a) - 0.1 / (2 * a)
    gt_root2 = -b / (2 * a) + 0.1 / (2 * a)
    
    ub = -b / (2 * a) - 0.2 / (2 * a)
    lb = -b / (2 * a) - 0.4 / (2 * a)

    fun_fn = lambda x: a * x ** 2 + b * x + c
    jac_fn = None  # lambda x: 2 * a * x + b

    assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(gt_root1))) <= D.tol_epsilon(dtype_var))
    assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(gt_root2))) <= D.tol_epsilon(dtype_var))

    for test_idx in range(4):
        x0 = lb + test_idx * (ub - lb) / 4

        root, (success, *_) = solver(fun_fn, x0, jac=jac_fn, tol=tolerance, verbose=1)

        assert (success)
        
        conv_root1 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root1), tol, tol)
        conv_root2 = np.allclose(D.ar_numpy.to_numpy(root), D.ar_numpy.to_numpy(gt_root2), tol, tol)
        
        assert D.ar_numpy.all(conv_root1 | conv_root2)
        assert D.ar_numpy.all(D.ar_numpy.to_numpy(D.ar_numpy.abs(fun_fn(root))) <= tol)


@pytest.mark.slow
@pytest.mark.parametrize('solver', [de.utilities.optimizer.nonlinear_roots, de.utilities.optimizer.newtontrustregion, de.utilities.optimizer.hybrj])
@common.test_fn_param
def test_rootfinding_robustness(fn, solver, dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    if backend_var == 'torch':
        set_torch_printoptions()
    
    tolerance, tol = convert_tolerance(None, dtype_var)

    x0 = D.ar_numpy.asarray(fn.root_interval[0] + 0.5 * (fn.root_interval[1] - fn.root_interval[0]), dtype=dtype_var, like=backend_var)

    root, (success, *_) = solver(fn, x0, jac=fn.jac, tol=tolerance, verbose=0)

    assert (success)
    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fn(root))) <= tol)
    
    root, (success, *_) = solver(fn, x0, jac=None, tol=tolerance, verbose=0)

    assert (success)
    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fn(root))) <= tol)
    
    root, (success, *_) = solver(fn, x0, jac=fn.jac, tol=tolerance, verbose=1, var_bounds=fn.root_interval)

    assert (success)
    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fn(root))) <= tol)
    
    root, (success, *_) = solver(fn, x0, jac=None, tol=tolerance, verbose=1, var_bounds=fn.root_interval)

    assert (success)
    assert (D.ar_numpy.to_numpy(D.ar_numpy.abs(fn(root))) <= tol)
