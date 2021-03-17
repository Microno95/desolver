# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run_implicit", action="store_true", default=False, help="run implicit integrator tests"
    )
    parser.addoption(
        "--run_explicit", action="store_true", default=False, help="run explicit integrator tests"
    )
    parser.addoption(
        "--run_torch_gradients", action="store_true", default=False, help="run torch gradients integrator tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "implicit: mark test as implicit integrator test")
    config.addinivalue_line("markers", "explicit: mark test as explicit integrator test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run_implicit"):
        # --run_implicit given in cli: do not skip implicit integrator tests
        pass
    else:
        skip_implicit = pytest.mark.skip(reason="need --run_implicit option to run")
        for item in items:
            if "implicit" in item.keywords:
                item.add_marker(skip_implicit)
    if config.getoption("--run_explicit"):
        # --run_explicit given in cli: do not skip explicit integrator tests
        pass
    else:
        skip_explicit = pytest.mark.skip(reason="need --run_explicit option to run")
        for item in items:
            if "explicit" in item.keywords:
                item.add_marker(skip_explicit)
    if config.getoption("--run_torch_gradients"):
        # --run_explicit given in cli: do not skip slow tests
        pass
    else:
        skip_torch_gradients = pytest.mark.skip(reason="need --run_torch_gradients option to run")
        for item in items:
            if "torch_gradients" in item.keywords:
                item.add_marker(skip_torch_gradients)