from .. import backend as D
import abc
import typing

__all__ = [
    'IntegratorTemplate',
    'RichardsonIntegratorTemplate'
]

class IntegratorTemplate(abc.ABC):
    symplectic = False
    order = None

    def __init__(self):
        self.solver_dict = None
        self.__custom_adaptation_fn = None

    @property
    def is_adaptive(self):
        return False

    @property
    def adaptation_fn(self):
        return self.__custom_adaptation_fn if self.__custom_adaptation_fn else self.update_timestep
    
    @adaptation_fn.setter
    def adaptation_fn(self, update_step_fn: typing.Callable):
        try:
            adaptation_fn_outputs = update_step_fn(self)
            assert isinstance(adaptation_fn_outputs, tuple)
            assert len(adaptation_fn_outputs) == 2
            assert isinstance(adaptation_fn_outputs[0], float)
            assert isinstance(adaptation_fn_outputs[1], bool)
        except AssertionError:
            raise ValueError("Step adaptation function must be a function that takes `self` and returns a tuple of exactly two outputs of the form: (timestep[float], redo_step[bool])")
        self.__custom_adaptation_fn = update_step_fn

    @abc.abstractmethod
    def __call__(self, rhs, initial_time, initial_state, constants, timestep):
        pass

    @abc.abstractmethod
    def dense_output(self):
        pass

    def update_timestep(self, ignore_custom_adaptation=False):
        if self.adaptation_fn and not ignore_custom_adaptation:
            return self.adaptation_fn(self)
        initial_state = self.solver_dict['initial_state']
        diff = self.solver_dict['diff']
        timestep = self.solver_dict['timestep']
        safety_factor = self.solver_dict['safety_factor']
        atol = self.solver_dict['atol']
        rtol = self.solver_dict['rtol']
        dState = self.solver_dict['dState']
        order = self.solver_dict['order']
        err_estimate = D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(diff)))
        relerr = D.ar_numpy.max(
            D.ar_numpy.to_numpy(atol + rtol * D.ar_numpy.abs(initial_state) + rtol * D.ar_numpy.abs(dState / timestep))
        )
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
        if self.device is not None:
            return "<{}({},{},{},{},{})>".format(self.__class__.__name__, self.dim, self.dtype, self.rtol, self.atol,
                                                 self.device)
        else:
            return "<{}({},{},{},{})>".format(self.__class__.__name__, self.dim, self.dtype, self.rtol, self.atol)


class RichardsonIntegratorTemplate(IntegratorTemplate, abc.ABC):
    symplectic = False

    @property
    def adaptive(self):
        return True
