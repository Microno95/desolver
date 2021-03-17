from .. import backend as D

__all__ = [
    'IntegratorTemplate',
    'RichardsonIntegratorTemplate'
]

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
            return "<{}({},{},{},{},{})>".format(self.__class__.__name__, self.dim, self.dtype, self.rtol, self.atol, self.device)
        else:
            return "<{}({},{},{},{})>".format(self.__class__.__name__, self.dim, self.dtype, self.rtol, self.atol)


class RichardsonIntegratorTemplate(IntegratorTemplate):
    __symplectic__ = False
    __adaptive__   = True

    def __init__(self):
        raise NotImplementedError("Do not initialise this class directly!")
        
    def dense_output(self):
        raise NotImplementedError("Do not use this class directly! How did you initialise it??")

    def adaptive_richardson(self, rhs, t, y, constants, timestep):
        dt0, (dt_z, dy_z) = self.step(0, rhs, t, y, timestep, constants, 1)
        if dt_z < timestep:
            timestep = dt_z
        self.aux[0, 0] = dy_z
        prev_error = None
        for m in range(1, self.richardson_iter):
            self.aux[m, 0] = self.step(m, rhs, t, y, timestep, constants, 1 << m)[1][1]
            for n in range(1, m+1):
                self.aux[m, n] = self.aux[m, n - 1] + (self.aux[m, n - 1] - self.aux[m - 1, n - 1]) / ((1 << n) - 1)
            self.order = self.basis_order + m + 1
            if m >= 3:
                prev_error, t_conv = self.check_converged(self.aux[m, n], self.aux[m - 1, m - 1] - self.aux[m, m], prev_error)
                if t_conv:
                    break

        return timestep, (timestep, self.aux[m - 1, n - 1]), self.aux[m - 1, m - 1] - self.aux[m, m]

    def check_converged(self, initial_state, diff, prev_error):
        err_estimate = D.max(D.abs(D.to_float(diff)))
        relerr       = D.max(D.to_float(self.atol + self.rtol * D.abs(initial_state)))
        if prev_error is None or (err_estimate > relerr and err_estimate <= D.max(D.abs(D.to_float(prev_error)))):
            return diff, False
        else:
            return diff, True

    def step(self, int_num, rhs, initial_time, initial_state, timestep, constants, num_intervals):
        dt_now, dstate_now = 0.0, 0.0
        dtstep = timestep / num_intervals
        for interval in range(num_intervals):
            dt, (dt_z, dy_z) = self.basis_integrators[int_num](rhs, initial_time + dt_now, initial_state + dstate_now, constants, dtstep)
            dt_now     = dt_now     + dt_z
            dstate_now = dstate_now + dy_z
        return dtstep, (dt_now, dstate_now)

    def forward(self, rhs, initial_time, initial_state, constants, timestep):
        dt0, (dt_z, dy_z), diff = self.adaptive_richardson(rhs, initial_time, initial_state, constants, timestep)

        self.dState = dy_z + 0.0
        self.dTime  = D.copy(dt_z)
        
        new_timestep, redo_step = self.update_timestep(initial_state, self.dState, diff, initial_time, dt_z, tol=0.5 if self.__implicit__ else 0.9)
        if self.__symplectic__:
            timestep = dt0
            uppers  = 0
            downers = 0
            if new_timestep < timestep:
                while new_timestep < timestep:
                    timestep /= 2.0
                    downers += 1
            else:
                while new_timestep > 2 * timestep:
                    timestep *= 2.0
                    uppers += 1
                redo_step = False
        else:
            timestep = new_timestep
        if redo_step:
            timestep, (self.dTime, self.dState) = self(rhs, initial_time, initial_state, constants, timestep)
            
        return timestep, (self.dTime, self.dState)

    __call__ = forward