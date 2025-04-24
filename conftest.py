# content of conftest.py

import pytest
import copy
import itertools
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from desolver.integrators import explicit_methods, implicit_methods, available_methods

def available_backends():
    available_backends = ["numpy"]
    try:
        import torch
        available_backends.append("torch")
    except ImportError:
        pass
    return available_backends

def available_device_var():
    available_device_var = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            available_device_var.append("cuda:0")
        torch.set_num_threads(1)
    except ImportError:
        pass
    return available_device_var


# Arrange
@pytest.fixture(scope='function', params=explicit_methods())
def explicit_integrators(request):
    return request.param


@pytest.fixture(scope='function', params=implicit_methods())
def implicit_integrators(request):
    return request.param


@pytest.fixture(scope='function', params=sorted(set(available_methods(False).values()), key=str))
def integrators(request):
    return request.param


@pytest.fixture(scope='function', params=[None] if "torch" in available_backends() else [])
def pytorch_only(request):
    return request.param


def pytest_generate_tests(metafunc: pytest.Metafunc):
    autodiff_needed = "requires_autodiff" in metafunc.fixturenames
    
    argvalues_map = {
        "dtype_var": ["float16", "float32", "float64"],
        "backend_var": available_backends(),
        "device_var": available_device_var()
    }
    
    argnames = sorted([key for key in argvalues_map if key in metafunc.fixturenames])
    argvalues = list(itertools.product(*[argvalues_map[key] for key in argnames]))
    
    if "dtype_var" and "backend_var" in metafunc.fixturenames:
        if np.finfo(np.longdouble).bits > np.finfo(np.float64).bits:
            expansion_map = {
                "dtype_var": ["longdouble"],
                "backend_var": ["numpy"]
            }
            if "device_var" in metafunc.fixturenames:
                expansion_map["device_var"] = [None]
            argvalues.extend(list(itertools.product(*[expansion_map[key] for key in argnames])))
        
        if "torch" in argvalues_map["backend_var"]:
            expansion_map = {
                "dtype_var": ["bfloat16"],
                "backend_var": ["torch"]
            }
            if "device_var" in metafunc.fixturenames:
                expansion_map["device_var"] = argvalues_map["device_var"]
            argvalues.extend(list(itertools.product(*[expansion_map[key] for key in argnames])))

    if autodiff_needed:
        if "backend_var" not in metafunc.fixturenames:
            raise TypeError("Test configuration requests autodiff, but no dynamic backend specified!")
        argnames.append("requires_autodiff")
        argvalues = [(*aval, True) for aval in argvalues if len(aval) > 1 and aval[1] not in ["numpy"]]
    
    metafunc.parametrize(argnames, argvalues)
