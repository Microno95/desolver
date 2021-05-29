from .. import backend as D
import abc

__all__ = [
    'IntegratorTemplate',
    'RichardsonIntegratorTemplate'
]


class IntegratorTemplate(abc.ABC):
    symplectic = False
    order = None

    def __init__(self):
        self.solver_dict = None

    @property
    def adaptive(self):
        return False

    @adaptive.setter
    def adaptive(self, adaptive):
        pass

    @abc.abstractmethod
    def __call__(self, rhs, initial_time, initial_state, constants, timestep):
        pass

    @abc.abstractmethod
    def dense_output(self):
        pass

    def update_timestep(self):
        initial_state = self.solver_dict['initial_state']
        diff = self.solver_dict['diff']
        timestep = self.solver_dict['timestep']
        safety_factor = self.solver_dict['safety_factor']
        atol = self.solver_dict['atol']
        rtol = self.solver_dict['rtol']
        dState = self.solver_dict['dState']
        order = self.solver_dict['order']
        err_estimate = D.max(D.abs(D.to_float(diff)))
        relerr = D.max(
            D.to_float(atol + rtol * D.abs(initial_state) + rtol * D.abs(dState / timestep)))
        corr = 1.0
        if err_estimate != 0:
            corr = corr * safety_factor * (relerr / err_estimate) ** (1.0 / order)
        if corr != 0:
            timestep = corr * timestep
        if err_estimate > relerr:
            return timestep, True
        else:
            return timestep, False

    def get_error_estimate(self):
        return 0.0

    @classmethod
    def __str__(cls):
        return cls.__name__

    def __repr__(self):
        if D.backend() == 'torch':
            return "<{}({},{},{},{},{})>".format(self.__class__.__name__, self.dim, self.dtype, self.rtol, self.atol,
                                                 self.device)
        else:
            return "<{}({},{},{},{})>".format(self.__class__.__name__, self.dim, self.dtype, self.rtol, self.atol)


class RichardsonIntegratorTemplate(IntegratorTemplate, abc.ABC):
    symplectic = False

    @property
    def adaptive(self):
        return True
