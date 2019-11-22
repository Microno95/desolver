from backend_test_utility import test_backend

import os
os.environ['DES_BACKEND'] = 'numpy'
import numpy as np

if __name__ == "__main__":
    np.testing.run_module_suite()