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
    def adaptation_fn(self):# -> Callable[..., Any] | Callable[..., Any | tuple[Any, Liter...:
        return self.__custom_adaptation_fn if self.__custom_adaptation_fn else self.update_timestep
    
    @adaptation_fn.setter
    def adaptation_fn(self, update_step_fn: typing.Callable):
        try:
            adaptation_fn_outputs = update_step_fn(self)
            assert isinstance(adaptation_fn_outputs, tuple)
            assert len(adaptation_fn_outputs) == 2
            a = 2.0 * adaptation_fn_outputs[0]
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
        if "system_scaling" in self.solver_dict:
            self.solver_dict["system_scaling"] = 0.8 * self.solver_dict["system_scaling"] +  0.2 * D.ar_numpy.maximum(D.ar_numpy.abs(initial_state), D.ar_numpy.abs(dState / timestep))
        else:
            self.solver_dict["system_scaling"] = D.ar_numpy.maximum(D.ar_numpy.abs(initial_state), D.ar_numpy.abs(dState / timestep))
        total_error_tolerance = (atol + rtol * self.solver_dict["system_scaling"])
        with D.numpy.errstate(divide='ignore'):
            epsilon_current = D.ar_numpy.reciprocal(D.ar_numpy.linalg.norm(diff / total_error_tolerance))
        if "epsilon_last" in self.solver_dict:
            epsilon_last = self.solver_dict["epsilon_last"]
        else:
            epsilon_last = None
        if "epsilon_last_last" in self.solver_dict:
            epsilon_last_last = self.solver_dict["epsilon_last_last"]
        else:
            epsilon_last_last = None
        if epsilon_last is None:
            corr = D.ar_numpy.where(epsilon_current > 0.0, epsilon_current ** (1.0 / order), 1.0)
            self.solver_dict["epsilon_last"] = epsilon_current
        elif epsilon_last is not None and epsilon_last_last is None:
            corr = D.ar_numpy.where(epsilon_current > 0.0, epsilon_current ** (1.0 / order), 1.0)
            corr = corr*D.ar_numpy.where(epsilon_last > 0.0, epsilon_last ** (1.0 / order), 1.0)
            self.solver_dict["epsilon_last_last"], self.solver_dict["epsilon_last"] = epsilon_last, epsilon_current
        elif epsilon_last is not None and epsilon_last_last is not None:
            # Based on the triple product described in https://link.springer.com/article/10.1007/s42967-021-00159-w
            # Eq. (6) with the coefficients from the second entry of Table 4
            k1, k2, k3 = self.solver_dict.get("__adapt_k1", 0.55), self.solver_dict.get("__adapt_k2", -0.27), self.solver_dict.get("__adapt_k3", 0.05)
            k1 = epsilon_current ** (k1 / order)
            k2 = epsilon_last ** (k2 / order)
            k3 = epsilon_last_last ** (k3 / order)
            corr = D.ar_numpy.where(k1 > 0.0, k1, 1.0)
            corr = corr*D.ar_numpy.where(k2 > 0.0, k2, 1.0)
            corr = corr*D.ar_numpy.where(k3 > 0.0, k3, 1.0)
            self.solver_dict["epsilon_last_last"], self.solver_dict["epsilon_last"] = epsilon_last, epsilon_current
        corr = (1 + D.ar_numpy.arctan((safety_factor * corr - 1)))
        timestep = corr * timestep
        return timestep, bool(corr < 0.9**2)

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
