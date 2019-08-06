from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

from .common import *

if 'DES_BACKEND' in os.environ:
    set_backend(os.environ['DES_BACKEND'])

if backend() == 'numpy':
    from .numpy_backend import *
elif backend() == 'torch':
    from .torch_backend import *
else:
    raise ValueError("Unable to import backend : " + str(backend()))
    
print("Using " + str(backend()) + " backend", file=sys.stderr)