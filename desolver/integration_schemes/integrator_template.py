"""
The MIT License (MIT)

Copyright (c) 2019 Microno95, Ekin Ozturk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from .. import backend as D

__all__ = [
    'IntegratorTemplate',
    'named_integrator',
]

def named_integrator(name, alt_names=tuple(), order=1.0):
    def wrap(f):
        f.__name__ = str(name)
        f.__alt_names__ = alt_names
        f.__order__ = order
        if hasattr(f, "final_state"):
            f.__adaptive__ = D.shape(f.final_state)[0] == 2
        else:
            f.__adaptive__ = False
        return f
    return wrap

class IntegratorTemplate(object):
    def __init__(self):
        raise NotImplementedError("Do not initialise this class directly!")

    def forward(self, rhs, initial_time, initial_state, constants, timestep):
        raise NotImplementedError("Do not use this class directly! How did you initialise it??")
        
    def dense_output(self):
        raise NotImplementedError("Do not use this class directly! How did you initialise it??")

    __call__ = forward

    def update_timestep(self, diff, initial_time, timestep, tol=0.9):
        err_estimate = D.max(D.abs(diff))
        relerr = self.atol + self.rtol * err_estimate
        if err_estimate != 0:
            corr = timestep * tol * (relerr / err_estimate) ** (1.0 / self.num_stages)
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

