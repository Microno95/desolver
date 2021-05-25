from .. import backend as D
import abc

__all__ = [
    'IntegratorTemplate',
    'RichardsonIntegratorTemplate'
]


class IntegratorTemplate(abc.ABC):
    symplectic = False

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

    @abc.abstractmethod
    def update_timestep(self, initial_state, diff, initial_time, timestep, tol):
        pass

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
