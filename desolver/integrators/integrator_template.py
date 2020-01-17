from .. import backend as D

__all__ = [
    'IntegratorTemplate',
#     'named_integrator',
]

# def named_integrator(name, alt_names=tuple(), order=1.0):
#     def wrap(f):
#         f.__name__ = str(name)
#         f.__alt_names__ = alt_names
#         if hasattr(f, 'order'):
#             f.__order__ = f.order
#         else:
#             f.__order__ = order
#         if hasattr(f, "final_state"):
#             f.__adaptive__ = D.shape(f.final_state)[0] == 2
#         else:
#             f.__adaptive__ = False
#         return f
#     return wrap

class IntegratorTemplate(object):
    def __init__(self):
        raise NotImplementedError("Do not initialise this class directly!")

    def forward(self, rhs, initial_time, initial_state, constants, timestep):
        raise NotImplementedError("Do not use this class directly! How did you initialise it??")
        
    def dense_output(self):
        raise NotImplementedError("Do not use this class directly! How did you initialise it??")

    __call__ = forward

    def update_timestep(self, initial_state, dState, diff, initial_time, timestep, tol=0.8):
        err_estimate = D.max(D.abs(D.to_float(diff)))
        relerr = D.max(D.to_float(self.atol + self.rtol * D.abs(initial_state) + self.rtol * D.abs(dState / timestep)))
        if err_estimate != 0:
            corr = timestep * tol * (relerr / err_estimate) ** (1.0 / self.order)
            if corr != 0:
                timestep = corr
        if err_estimate > relerr:
            return timestep, True
        else:
            return timestep, False
    
    @classmethod
    def __str__(cls):
        return cls.__name__
    
    def __repr__(self):
        if D.backend() == 'torch':
            return "<{}({},{},{},{})>".format(self.__class__.__name__, self.dim, self.dtype, self.rtol, self.atol)
        else:
            return "<{}({},{},{},{},{})>".format(self.__class__.__name__, self.dim, self.dtype, self.rtol, self.atol, self.device)

