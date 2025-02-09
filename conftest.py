# content of conftest.py

import pytest
import copy
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


def pytest_generate_tests(metafunc: pytest.Metafunc):
    available_dtypes = ["float16", "float32", "float64"]
    autodiff_needed = "requires_autodiff" in metafunc.fixturenames
    argnames = []
    argvalues = []
    if "dtype_var" in metafunc.fixturenames:
        argnames.append("dtype_var")
        if "backend_var" not in metafunc.fixturenames:
            argvalues = available_dtypes
        else:
            argvalues = []
            argnames.append("backend_var")
            for backend in available_backends():
                argvalues.extend([(dtype, backend) for dtype in available_dtypes])
                match backend:
                    case "numpy":
                        argvalues.append(("longdouble", "numpy"))
                    case "torch":
                        argvalues.append(("bfloat16", "torch"))
            if "device_var" in metafunc.fixturenames:
                argnames.append("device_var")
                argvalues_old = copy.deepcopy(argvalues)
                argvalues = []
                for dtype, backend in argvalues_old:
                    if backend == "numpy":
                        argvalues.append((dtype, backend, None))
                    else:
                        for device in available_device_var():
                            if device == "cpu":
                                argvalues.append((dtype, backend, device))
                            else:
                                argvalues.append((dtype, backend, device))
        if autodiff_needed:
            if "backend_var" not in metafunc.fixturenames:
                raise TypeError("Test configuration requests autodiff, but no dynamic backend specified!")
            argnames.append("requires_autodiff")
            argvalues = [(*aval, True) for aval in argvalues if len(aval) > 1 and aval[1] not in ["numpy"]]
        metafunc.parametrize(argnames, argvalues)
