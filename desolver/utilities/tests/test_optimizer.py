import desolver as de
import desolver.backend as D
import numpy as np
import pytest


def test_brentsroot_same_sign():
    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(True)

    ac_prod = D.array(np.random.uniform(0.9, 1.1))
    a = D.array(1.0)
    b = D.array(1.0)

    gt_root = -b / a
    lb, ub = -b / a - 1, -b / a - 2

    fun = lambda x: a * x + b

    assert (D.to_numpy(D.to_float(D.abs(fun(gt_root)))) <= 32 * D.epsilon())

    root, success = de.utilities.optimizer.brentsroot(fun, [lb, ub], 4 * D.epsilon(), verbose=True)

    assert (np.isinf(root))
    assert (not success)


def test_brentsroot_wrong_order():
    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(True)

    a = D.array(1.0)
    b = D.array(1.0)

    gt_root = -b / a
    lb, ub = -b / a - 1, -b / a + 1

    fun = lambda x: a * x + b

    assert (D.to_numpy(D.to_float(D.abs(fun(gt_root)))) <= 32 * D.epsilon())

    root, success = de.utilities.optimizer.brentsroot(fun, [ub, lb], 4 * D.epsilon(), verbose=True)

    assert (success)
    assert (
        np.allclose(D.to_numpy(D.to_float(gt_root)), D.to_numpy(D.to_float(root)), 32 * D.epsilon(), 32 * D.epsilon()))


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('tol', [None, 4, 1e-1])
def test_brentsroot(ffmt, tol):
    print("Set dtype to:", ffmt)
    D.set_float_fmt(ffmt)

    if tol is not None:
        tol = tol * D.epsilon()

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(True)

    for _ in range(10):
        ac_prod = D.array(np.random.uniform(0.9, 1.1))
        a = D.array(np.random.uniform(-1, 1))
        a = D.to_float(-1 * (a <= 0) + 1 * (a > 0))
        c = ac_prod / a
        b = D.sqrt(0.01 + 4 * ac_prod)

        gt_root = -b / (2 * a) - 0.1 / (2 * a)

        ub = -b / (2 * a)
        lb = -b / (2 * a) - 1.0 / (2 * a)

        fun = lambda x: a * x ** 2 + b * x + c

        assert (D.to_numpy(D.to_float(D.abs(fun(gt_root)))) <= 32 * D.epsilon())

        root, success = de.utilities.optimizer.brentsroot(fun, [lb, ub], tol, verbose=True)

        assert (success)
        assert (np.allclose(D.to_numpy(D.to_float(gt_root)), D.to_numpy(D.to_float(root)), 32 * D.epsilon(),
                            32 * D.epsilon()))
        assert (D.to_numpy(D.to_float(D.abs(fun(root)))) <= 32 * D.epsilon())


@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('tol', [None, 4, 1e-1])
def test_brentsrootvec(ffmt, tol):
    print("Set dtype to:", ffmt)
    D.set_float_fmt(ffmt)
    if tol is not None:
        tol = tol * D.epsilon()

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(True)

    if ffmt == 'gdual_vdouble':
        pytest.skip("Root-finding is ill-conceived with vectorised gduals")

    for _ in range(10):
        slope_list = D.array(np.copysign(np.random.uniform(0.9, 1.1, size=25), np.random.uniform(-1, 1, size=25)))
        intercept_list = slope_list

        gt_root_list = -intercept_list / slope_list

        fun_list = [(lambda m, b: lambda x: m * x + b)(m, b) for m, b in zip(slope_list, intercept_list)]

        assert (all(map((lambda i: D.to_numpy(D.to_float(D.abs(i))) <= 32 * D.epsilon()),
                        map((lambda x: x[0](x[1])), zip(fun_list, gt_root_list)))))

        root_list, success = de.utilities.optimizer.brentsrootvec(fun_list,
                                                                  [D.min(gt_root_list) - 1., D.max(gt_root_list) + 1.],
                                                                  tol, verbose=True)

        assert (np.all(D.to_numpy(success)))
        assert (np.allclose(D.to_numpy(D.to_float(gt_root_list)), D.to_numpy(D.to_float(root_list)), 32 * D.epsilon(),
                            32 * D.epsilon()))

        assert (all(map((lambda i: D.to_numpy(D.to_float(D.abs(i))) <= 32 * D.epsilon()),
                        map((lambda x: x[0](x[1])), zip(fun_list, root_list)))))


@pytest.mark.parametrize('solver', [de.utilities.optimizer.newtontrustregion, de.utilities.optimizer.nonlinear_roots])
@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('tol', [None, 40, 1])
def test_nonlinear_root(solver, ffmt, tol):
    print("Set dtype to:", ffmt)
    D.set_float_fmt(ffmt)
    np.random.seed(30)

    if tol is not None:
        tol = tol * D.epsilon()

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(False)

    if ffmt == 'gdual_vdouble':
        pytest.skip("Root-finding is ill-conceived with vectorised gduals")

    for _ in range(10):
        ac_prod = D.array(np.random.uniform(0.9, 1.1))
        a = D.array(np.random.uniform(-1, 1))
        a = D.to_float(-1 * (a <= 0) + 1 * (a > 0))
        c = ac_prod / a
        b = D.sqrt(0.01 + 4 * ac_prod)

        gt_root1 = -b / (2 * a) - 0.1 / (2 * a)
        gt_root2 = -b / (2 * a) + 0.1 / (2 * a)

        ub = -b / (2 * a) - 0.2 / (2 * a)
        lb = -b / (2 * a) - 0.4 / (2 * a)

        x0 = D.array(np.random.uniform(ub, lb))

        fun = lambda x: a * x ** 2 + b * x + c
        jac = lambda x: 2 * a * x + b

        assert (D.to_numpy(D.to_float(D.abs(fun(gt_root1)))) <= 32 * D.epsilon())
        assert (D.to_numpy(D.to_float(D.abs(fun(gt_root2)))) <= 32 * D.epsilon())

        root, (success, num_iter, nfev, njev, prec) = solver(fun, x0, jac=jac, tol=tol, verbose=True)

        if tol is None:
            tol = D.epsilon()
        conv_root1 = np.allclose(D.to_numpy(D.to_float(gt_root1)), D.to_numpy(D.to_float(root)), 128 * tol, 32 * tol)
        conv_root2 = np.allclose(D.to_numpy(D.to_float(gt_root2)), D.to_numpy(D.to_float(root)), 128 * tol, 32 * tol)
        print(conv_root1, conv_root2, root, gt_root1, gt_root2, x0, root - gt_root1, root - gt_root2, num_iter, prec)

        assert (success)
        assert (conv_root1 or conv_root2)
        assert (D.to_numpy(D.to_float(D.abs(fun(root)))) <= 32 * tol)


@pytest.mark.parametrize('solver', [de.utilities.optimizer.newtontrustregion,de.utilities.optimizer.nonlinear_roots])
@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('tol', [None, 40, 1])
def test_nonlinear_root_estimated_jac(solver, ffmt, tol):
    print("Set dtype to:", ffmt)
    D.set_float_fmt(ffmt)
    np.random.seed(20)

    if tol is not None:
        tol = tol * D.epsilon()

    if ffmt == 'gdual_vdouble':
        pytest.skip("Root-finding is ill-conceived with vectorised gduals")

    for _ in range(10):
        ac_prod = D.array(np.random.uniform(0.9, 1.1))
        a = D.array(np.random.uniform(-1, 1))
        a = D.to_float(-1 * (a <= 0) + 1 * (a > 0))
        c = ac_prod / a
        b = D.sqrt(0.01 + 4 * ac_prod)

        gt_root1 = -b / (2 * a) - 0.1 / (2 * a)
        gt_root2 = -b / (2 * a) + 0.1 / (2 * a)

        ub = -b / (2 * a) - 0.2 / (2 * a)
        lb = -b / (2 * a) - 0.4 / (2 * a)

        x0 = D.array(np.random.uniform(ub, lb))

        fun = lambda x: a * x ** 2 + b * x + c
        jac = de.utilities.JacobianWrapper(fun)

        assert (D.to_numpy(D.to_float(D.abs(fun(gt_root1)))) <= 32 * D.epsilon())
        assert (D.to_numpy(D.to_float(D.abs(fun(gt_root2)))) <= 32 * D.epsilon())

        root, (success, num_iter, nfev, njev, prec) = solver(fun, x0, jac=jac, tol=tol, verbose=True)

        if tol is None:
            tol = D.epsilon()
        conv_root1 = np.allclose(D.to_numpy(D.to_float(gt_root1)), D.to_numpy(D.to_float(root)), 128 * tol, 32 * tol)
        conv_root2 = np.allclose(D.to_numpy(D.to_float(gt_root2)), D.to_numpy(D.to_float(root)), 128 * tol, 32 * tol)
        print(conv_root1, conv_root2, root, gt_root1, gt_root2, x0, root - gt_root1, root - gt_root2, num_iter, prec)

        assert (success)
        assert (conv_root1 or conv_root2)
        assert (D.to_numpy(D.to_float(D.abs(fun(root)))) <= 32 * tol)


@pytest.mark.parametrize('solver', [de.utilities.optimizer.newtontrustregion,de.utilities.optimizer.nonlinear_roots])
@pytest.mark.parametrize('ffmt', [i for i in D.available_float_fmt() if i != 'float16'])
@pytest.mark.parametrize('tol', [None, 40, 1])
@pytest.mark.parametrize('dim', [1, 5, 10, 100, 250])
def test_nonlinear_root_dims(solver, ffmt, tol, dim):
    if "gdual" in ffmt and dim > 100:
        pytest.skip("Optimizing high dimensional systems with gduals is too slow")
    print("Set dtype to:", ffmt)
    D.set_float_fmt(ffmt)
    np.random.seed(30)

    if tol is not None:
        tol = tol * D.epsilon()

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(False)

    if ffmt == 'gdual_vdouble':
        pytest.skip("Root-finding is ill-conceived with vectorised gduals")

    shift = D.array(np.random.uniform(1, 10, size=(dim,)))
    exponent = D.array(np.random.randint(1, 5, size=(dim,)))
    gt_root1 = shift ** (1 / exponent)
    gt_root2 = -shift ** (1 / exponent)

    def fun(x):
        return x ** exponent - shift

    def jac(x):
        return D.diag(exponent * D.reshape(x, (-1,)) ** (exponent - 1))

    x0 = D.array(np.random.uniform(1, 3, size=(dim,)))
    print(gt_root1, gt_root2)
    print(fun(gt_root1), fun(gt_root2))
    print(x0)
    print(fun(x0))
    print(jac(x0))

    root, (success, num_iter, nfev, njev, prec) = solver(fun, x0, jac=jac, tol=tol, verbose=True)

    root = root.reshape(x0.shape)

    if tol is None:
        tol = D.epsilon()
    assert (success)
    conv_root1 = D.stack(
        [D.array(np.allclose(D.to_numpy(D.to_float(r1)), D.to_numpy(D.to_float(r)), 128 * tol, 32 * tol), dtype=D.bool)
         for r, r1 in zip(root, gt_root1)])
    conv_root2 = D.stack(
        [D.array(np.allclose(D.to_numpy(D.to_float(r2)), D.to_numpy(D.to_float(r)), 128 * tol, 32 * tol), dtype=D.bool)
         for r, r2 in zip(root, gt_root2)])
    print(root)
    print(conv_root1)
    print(conv_root2)
    assert (D.all(conv_root1 | conv_root2))


@pytest.mark.skipif(D.backend() != 'torch', reason="Pytorch backend required to test jacobian via AD")
@pytest.mark.parametrize('solver', [de.utilities.optimizer.newtontrustregion,de.utilities.optimizer.nonlinear_roots])
@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('tol', [None, 40, 1])
def test_nonlinear_root_pytorch_jacobian(solver, ffmt, tol):
    print("Set dtype to:", ffmt)
    D.set_float_fmt(ffmt)
    np.random.seed(21)

    if tol is not None:
        tol = tol * D.epsilon()

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(False)

    if ffmt == 'gdual_vdouble':
        pytest.skip("Root-finding is ill-conceived with vectorised gduals")

    for _ in range(10):
        ac_prod = D.array(np.random.uniform(0.9, 1.1))
        a = D.array(np.random.uniform(-1, 1))
        a = D.to_float(-1 * (a <= 0) + 1 * (a > 0))
        c = ac_prod / a
        b = D.sqrt(0.01 + 4 * ac_prod)

        gt_root1 = -b / (2 * a) - 0.1 / (2 * a)
        gt_root2 = -b / (2 * a) + 0.1 / (2 * a)

        ub = -b / (2 * a) - 0.2 / (2 * a)
        lb = -b / (2 * a) - 0.4 / (2 * a)

        x0 = D.array(np.random.uniform(ub, lb))

        fun = lambda x: a * x ** 2 + b * x + c

        assert (D.to_numpy(D.to_float(D.abs(fun(gt_root1)))) <= 32 * D.epsilon())
        assert (D.to_numpy(D.to_float(D.abs(fun(gt_root2)))) <= 32 * D.epsilon())

        root, (success, num_iter, nfev, njev, prec) = solver(fun, x0, tol=tol, verbose=True)

        if tol is None:
            tol = D.epsilon()
        conv_root1 = np.allclose(D.to_numpy(D.to_float(gt_root1)), D.to_numpy(D.to_float(root)), 128 * tol, 32 * tol)
        conv_root2 = np.allclose(D.to_numpy(D.to_float(gt_root2)), D.to_numpy(D.to_float(root)), 128 * tol, 32 * tol)
        print(conv_root1, conv_root2, root, gt_root1, gt_root2, x0, root - gt_root1, root - gt_root2, num_iter, prec)

        assert (success)
        assert (conv_root1 or conv_root2)
        assert (D.to_numpy(D.to_float(D.abs(fun(root)))) <= 32 * tol)


@pytest.mark.skipif(D.backend() != 'torch', reason="Pytorch backend required to test jacobian via AD")
@pytest.mark.parametrize('solver', [de.utilities.optimizer.newtontrustregion,de.utilities.optimizer.nonlinear_roots])
@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('tol', [None, 40, 1])
@pytest.mark.parametrize('dim', [1, 5, 10, 100, 250])
def test_nonlinear_root_dims_pytorch_jacobian(solver, ffmt, tol, dim):
    print("Set dtype to:", ffmt)
    D.set_float_fmt(ffmt)
    np.random.seed(30)

    if tol is not None:
        tol = tol * D.epsilon()

    if D.backend() == 'torch':
        import torch

        torch.set_printoptions(precision=17)

        torch.autograd.set_detect_anomaly(False)

    if ffmt == 'gdual_vdouble':
        pytest.skip("Root-finding is ill-conceived with vectorised gduals")

    shift = D.array(np.random.uniform(1, 10, size=(dim,)))
    exponent = D.array([2] * dim)
    gt_root1 = shift ** (1 / exponent)
    gt_root2 = -shift ** (1 / exponent)

    def fun(x):
        return x ** exponent - shift

    x0 = D.array(gt_root1) + D.array(np.random.uniform(-1e-4, 1e-4, size=(dim,)))
    print(gt_root1, gt_root2)
    print(x0)
    print(fun(x0))

    root, (success, num_iter, nfev, njev, prec) = solver(fun, x0, jac=None, tol=tol, verbose=True, maxiter=200)

    if tol is None:
        tol = D.epsilon()
    assert (success)
    conv_root1 = D.stack(
        [D.array(np.allclose(D.to_numpy(D.to_float(r1)), D.to_numpy(D.to_float(r)), 128 * tol, 32 * tol), dtype=D.bool)
         for r, r1 in zip(root, gt_root1)])
    conv_root2 = D.stack(
        [D.array(np.allclose(D.to_numpy(D.to_float(r2)), D.to_numpy(D.to_float(r)), 128 * tol, 32 * tol), dtype=D.bool)
         for r, r2 in zip(root, gt_root2)])
    assert (D.all(conv_root1 | conv_root2))
