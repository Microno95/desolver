from .symbolic import *
import os
import sys

def get_script_path():
    return os.path.dirname(os.path.realpath(__file__))
    
if not os.path.isfile(os.path.join(get_script_path(), "symbolic_funcs.py")):
    from .gen_auto import main as main
    main()
    del main

from .symbolic_funcs import *
