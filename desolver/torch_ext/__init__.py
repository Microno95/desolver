try:
    import torch
    from desolver.torch_ext.integrators import torch_solve_ivp
except ImportError:
    pass
