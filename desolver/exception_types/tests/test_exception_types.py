import desolver as de
import numpy as np


def test_recursion_error():
    try:
        raise de.exception_types.RecursionError()
    except de.exception_types.RecursionError:
        pass
    except:
        raise


def test_failed_integration_error():
    try:
        raise de.exception_types.FailedIntegrationError()
    except de.exception_types.FailedIntegrationError:
        pass
    except:
        raise


if __name__ == "__main__":
    np.testing.run_module_suite()
