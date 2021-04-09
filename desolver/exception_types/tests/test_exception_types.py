import desolver as de
import numpy as np


def test_recursion_error():
    try:
        raise de.exception_types.RecursionError()
    except de.exception_types.RecursionError:
        pass
    except:
        raise


def test_failed_integration():
    try:
        raise de.exception_types.FailedIntegration()
    except de.exception_types.FailedIntegration:
        pass
    except:
        raise

        
def test_tolerance_failure():
    try:
        raise de.exception_types.FailedToMeetTolerances()
    except de.exception_types.FailedToMeetTolerances:
        pass
    except:
        raise