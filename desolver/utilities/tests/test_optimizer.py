import desolver as de
import desolver.backend as D
import numpy as np
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

    root, success = de.utilities.optimizer.brentsroot(fun, [lb, ub], D.tol_epsilon(dtype_var), verbose=1)

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

    root, success = de.utilities.optimizer.brentsroot(fun, [lb, ub], D.tol_epsilon(dtype_var), verbose=1)

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
                                                                tolerance, verbose=1)

    assert (np.all(D.ar_numpy.to_numpy(success)))
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
