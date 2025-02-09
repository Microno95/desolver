import pytest
import numpy as np
import desolver.backend as D


def test_contract_first_ndims_case_1(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    arr1 = D.ar_numpy.array([[2.0, 1.0], [1.0, 0.0]], dtype=dtype_var, like=backend_var)
    arr2 = D.ar_numpy.array([[1.0, 1.0], [-1.0, 1.0]], dtype=dtype_var, like=backend_var)

    arr3 = D.contract_first_ndims(arr1, arr2, 1)

    true_arr3 = D.ar_numpy.array([1.0, 1.0], dtype=dtype_var, like=backend_var)

    assert (D.ar_numpy.linalg.norm(arr3 - true_arr3) <= 2 * D.epsilon(dtype_var))


def test_contract_first_ndims_case_2(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    arr1 = D.ar_numpy.array([[2.0, 1.0], [1.0, 0.0]], dtype=dtype_var, like=backend_var)
    arr2 = D.ar_numpy.array([[1.0, 1.0], [-1.0, 1.0]], dtype=dtype_var, like=backend_var)

    arr4 = D.contract_first_ndims(arr1, arr2, 2)

    true_arr4 = D.ar_numpy.array(2., dtype=dtype_var, like=backend_var)

    assert (D.ar_numpy.linalg.norm(arr4 - true_arr4) <= 2 * D.epsilon(dtype_var))


def test_contract_first_ndims_reverse_order(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    a = D.ar_numpy.array([1.0, 2.0], dtype=dtype_var, like=backend_var)
    b = D.ar_numpy.array([[1.0, 2.0], [2.0, 3.0]], dtype=dtype_var, like=backend_var)
    D.contract_first_ndims(b, a, n=1)


def test_contract_first_ndims_shape_too_small(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    with pytest.raises(ValueError):
        a = D.ar_numpy.array([1.0, 2.0], dtype=dtype_var, like=backend_var)
        b = D.ar_numpy.array([[1.0, 2.0], [2.0, 3.0]], dtype=dtype_var, like=backend_var)
        D.contract_first_ndims(a, b, n=2)


def test_epsilon(dtype_var, backend_var):
    dtype_var = D.autoray.to_backend_dtype(dtype_var, like=backend_var)
    finfo = D.ar_numpy.finfo(dtype_var)
    assert finfo.eps*4 == D.epsilon(dtype_var)
    assert finfo.eps*32 == D.tol_epsilon(dtype_var)