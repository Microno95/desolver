from .integrator_template import IntegratorTemplate, RichardsonIntegratorTemplate
from .. import backend as D
from .. import utilities
from .. import exception_types
import math
import builtins
import abc

__all__ = [
    'RungeKuttaIntegrator',
    'ExplicitSymplecticIntegrator',
    'generate_richardson_integrator'
]


class TableauIntegrator(IntegratorTemplate, abc.ABC):
    tableau = None
    __order__ = None
    def __init__(self, sys_dim, dtype=None, rtol=None, atol=None, device=None):
        super().__init__()
        self.dim = sys_dim
        self.numel = 1
        for i in self.dim:
            self.numel *= int(i)
        self.rtol = rtol if rtol is not None else 32 * D.epsilon()
        self.atol = atol if atol is not None else 32 * D.epsilon()
        self.dtype = dtype
        self.dState = D.zeros(self.dim, dtype=self.dtype)
        self.dTime = D.zeros(tuple(), dtype=self.dtype)
        self.device = device
        self.initial_state = None
        self.initial_rhs = None
        self.initial_time = None
        self.final_rhs = None
        self.__explicit = None
        self.__adaptive = False
        self.__fsal = False
        if dtype is None:
            self.tableau = D.array(self.tableau)
        else:
            self.tableau = D.to_type(self.tableau, dtype)
        if D.backend() == 'torch':
            self.dState = self.dState.to(self.device)
            self.dTime = self.dTime.to(self.device)
            self.tableau = self.tableau.to(self.device)

    # Class properties for accessing attributes of the class #
    @classmethod
    def integrator_order(cls):
        return cls.__order__
    # ---- #

    # Instance properties that are cached #
    @property
    def order(self):
        return self.__class__.__order__

    @property
    def stages(self):
        return self.tableau.shape[0]
    # ---- #

    @abc.abstractmethod
    def step(self, rhs, initial_time, initial_state, constants, timestep):
        pass


class RungeKuttaIntegrator(TableauIntegrator):
    final_state = None

    def __init__(self, sys_dim, dtype=None, rtol=None, atol=None, device=None):
        super().__init__(sys_dim=sys_dim, dtype=dtype, rtol=rtol, atol=atol, device=device)
        if dtype is None:
            self.final_state = D.array(self.final_state)
        else:
            self.final_state = D.to_type(self.final_state, dtype)
        if D.backend() == 'torch':
            self.final_state = self.final_state.to(self.device)
        self.__adaptive = self.__class__.is_adaptive()
        self.__fsal = self.__class__.is_fsal()
        self.__explicit = self.__class__.is_explicit()
        self.__explicit_stages = [col for col in range(self.stages) if D.all(self.tableau[col, col + 1:] == 0.0)]
        self.__implicit_stages = [col for col in range(self.stages) if D.any(self.tableau[col, col + 1:] != 0.0)]

        self.aux = D.zeros((self.stages, *self.dim), dtype=self.dtype)
        if D.backend() == "torch":
            self.aux = self.aux.to(self.device)

        self.tableau_idx_expand = tuple([slice(1, None, None)] + [None] * (self.aux.ndim - 1))
        self.solver_dict = dict(safety_factor=0.8, order=self.order, atol=self.atol, rtol=self.rtol)
        if self.implicit:
            self.solver_dict.update(dict(
                tau0=0, tau1=0, niter0=0, niter1=0
            ))

    # Class properties for accessing attributes of the class #
    @classmethod
    def is_adaptive(cls):
        return cls.final_state.shape[0] == 2

    @classmethod
    def is_explicit(cls):
        return all([(cls.tableau[col, col + 1:] == 0.0).all() for col in range(cls.tableau.shape[0])])

    @classmethod
    def is_implicit(cls):
        return any([(cls.tableau[col, col + 1:] != 0.0).any() for col in range(cls.tableau.shape[0])])

    @classmethod
    def is_fsal(cls):
        return bool((cls.tableau[-1, 1:] == cls.final_state[0][1:]).all())
    # ---- #

    # Instance properties that are cached #
    @property
    def adaptive(self):
        return self.__adaptive

    @adaptive.setter
    def adaptive(self, other):
        self.__adaptive = bool(other) & (D.shape(self.final_state)[0] == 2)

    @property
    def fsal(self):
        return self.__fsal

    @property
    def explicit(self):
        return self.__explicit

    @property
    def implicit(self):
        return not self.__explicit

    @property
    def explicit_stages(self):
        return self.__explicit_stages

    @property
    def implicit_stages(self):
        return self.__implicit_stages
    # ---- #

    def __call__(self, rhs, initial_time, initial_state, constants, timestep):
        self.initial_state = D.copy(initial_state)
        self.initial_time = D.copy(initial_time)
        self.initial_rhs = None
        if not self.fsal:
            self.final_rhs = None
        elif self.fsal and self.final_rhs is not None:
            self.aux[0] = self.final_rhs

        self.step(rhs=rhs, initial_time=initial_time, initial_state=initial_state,
                  constants=constants, timestep=timestep)

        if self.adaptive or self.implicit:
            self.solver_dict['diff'] = timestep * self.get_error_estimate()
            self.solver_dict['initial_state'] = initial_state
            self.solver_dict['initial_time'] = initial_time
            self.solver_dict['timestep'] = self.dTime
            self.solver_dict['atol'] = self.atol
            self.solver_dict['rtol'] = self.rtol
            self.solver_dict['dState'] = self.dState
            timestep, redo_step = self.update_timestep()
            if redo_step:
                for _ in range(64):
                    timestep, (self.dTime, self.dState) = self.step(rhs, initial_time, initial_state, constants,
                                                                    timestep)
                    self.solver_dict['diff'] = timestep * self.get_error_estimate()
                    self.solver_dict['timestep'] = self.dTime
                    self.solver_dict['dState'] = self.dState
                    timestep, redo_step = self.update_timestep()
                    if not redo_step:
                        break
                if redo_step:
                    raise exception_types.FailedToMeetTolerances(
                        "Failed to integrate system from {} to {} ".format(self.dTime, self.dTime + timestep) +
                        "to the tolerances required: rtol={}, atol={}".format(self.rtol, self.atol)
                    )

        if self.initial_rhs is None:
            self.initial_rhs = rhs(initial_time, initial_state, **constants)

        if self.final_rhs is None:
            self.final_rhs = rhs(initial_time + self.dTime, initial_state + self.dState, **constants)

        return timestep, (self.dTime, self.dState)

    def dense_output(self):
        return (self.initial_time + self.dTime,
                utilities.interpolation.CubicHermiteInterp(
                    self.initial_time,
                    self.initial_time + self.dTime,
                    self.initial_state,
                    self.initial_state + self.dState,
                    self.initial_rhs,
                    self.final_rhs
                ))

    def update_timestep(self):
        timestep = self.solver_dict['timestep']
        safety_factor = self.solver_dict['safety_factor']
        timestep_from_error, redo_step = super().update_timestep()
        if "niter0" in self.solver_dict.keys():
            if self.solver_dict['niter0'] != 0:
                Tk0, CTk0 = D.log(self.solver_dict['tau0']), math.log(self.solver_dict['niter0'])
                Tk1, CTk1 = D.log(self.solver_dict['tau1']), math.log(self.solver_dict['niter1'])
                dnCTk = D.array(CTk1 - CTk0)
                ddCTk = D.array(Tk1 - Tk0)
                if ddCTk > 0:
                    dCTk = dnCTk / ddCTk
                else:
                    dCTk = D.array(0.0)
                tau2 = timestep * D.exp(-safety_factor * dCTk)
            else:
                tau2 = timestep
            if tau2 < timestep_from_error:
                return tau2, redo_step
            else:
                return timestep_from_error, redo_step
        else:
            return timestep_from_error, redo_step

    def step(self, rhs, initial_time, initial_state, constants, timestep):
        aux_shape = self.aux.shape

        def nfun(next_state):
            nonlocal initial_time, initial_state, timestep, aux_shape
            __aux_states = D.reshape(next_state, aux_shape)
            __rhs_states = D.stack([
                rhs(initial_time + tbl[0] * timestep,
                    initial_state + timestep * D.sum(tbl[self.tableau_idx_expand] * __aux_states, axis=0), **constants)
                for
                tbl in self.tableau
            ])
            __states = D.reshape(__aux_states - __rhs_states, (-1,))
            return __states

        def __nfun_jac(next_state):
            nonlocal self, initial_time, initial_state, timestep, aux_shape
            __aux_states = D.reshape(next_state, aux_shape)
            __step = self.numel
            if D.backend() == 'torch':
                __jac = D.eye(self.tableau.shape[0] * __step, device=__aux_states.device, dtype=__aux_states.dtype)
            else:
                __jac = D.eye(self.tableau.shape[0] * __step)
            __prev_idx = -1
            __rhs_jac = D.stack([
                rhs.jac(initial_time + tbl[0] * timestep,
                        initial_state + timestep * D.sum(tbl[self.tableau_idx_expand] * __aux_states, axis=0),
                        **constants)
                for tbl in self.tableau
            ])
            for idx in range(0, __jac.shape[0], __step):
                for jdx in range(0, __jac.shape[1], __step):
                    __jac[idx:idx + __step, jdx:jdx + __step] -= timestep * self.tableau[
                        idx // __step, 1 + jdx // __step] * __rhs_jac[idx // __step].reshape(__step, __step)
            if __jac.shape[0] == 1 and __jac.shape[1] == 1:
                __jac = D.reshape(__jac, tuple())
            return __jac

        # Initial guess using a conservative midpoint guess #
        # initial_guess = D.copy(self.aux)
        #
        # midpoint_guess = D.stack([rhs(initial_time, initial_state, **constants)]*len(self.tableau))
        # self.initial_rhs = midpoint_guess[0]
        # midpoint_guess = D.stack([
        #     0.5 * timestep * tbl[0] * (md + rhs(initial_time + 0.5 * timestep * tbl[0],
        #                                         initial_state + 0.5 * timestep * tbl[0] * md,
        #                                         **constants)) for tbl, md in zip(self.tableau, midpoint_guess)
        # ])
        #
        # initial_guess = initial_guess + (0.5 * midpoint_guess + 0.5 * self.dState[None])

        # Initial guess from assuming method is explicit #
        current_state = None
        for stage in range(self.stages):
            current_state = initial_state + timestep * D.sum(self.tableau[stage][self.tableau_idx_expand] * self.aux,
                                                             axis=0)

            if stage > 0 or self.final_rhs is None:
                self.aux[stage] = rhs(initial_time + self.tableau[stage, 0] * timestep, current_state, **constants)

            # print("Current State:", current_state)
            # print("Auxiliary states:", self.aux)

            if stage == 0 and D.sum(self.tableau[0]) == 0.0:
                self.initial_rhs = self.aux[0]
            elif self.fsal and stage == self.stages - 1:
                self.final_rhs = self.aux[stage]

        if self.implicit:
            initial_guess = self.aux

            if rhs.jac_is_wrapped_rhs and D.backend() == 'torch':
                nfun_jac = None
                initial_guess.requires_grad = True
            else:
                nfun_jac = __nfun_jac

            desired_tol = D.max(D.abs(self.atol * 1e-1 + D.max(D.abs(D.to_float(self.rtol * 1e-1 * initial_state)))))
            aux_root, (success, num_iter, _, _, prec) = utilities.optimizer.nonlinear_roots(nfun, initial_guess,
                                                                                            jac=nfun_jac, verbose=False,
                                                                                            tol=desired_tol, maxiter=30)

            if not success and prec > desired_tol:
                raise exception_types.FailedToMeetTolerances(
                    "Step size too large, cannot solve system to the "
                    "tolerances required: achieved = {}, desired = {}, iter = {}".format(prec, desired_tol, num_iter))
            self.solver_dict.update(dict(
                tau0=self.solver_dict['tau1'], tau1=timestep,
                niter0=self.solver_dict['niter0'], niter1=num_iter
            ))

            self.aux = D.reshape(aux_root, aux_shape)

        if self.fsal and self.explicit:
            self.dState = current_state - initial_state
        else:
            self.dState = timestep * D.sum(self.final_state[0][self.tableau_idx_expand] * self.aux, axis=0)
        self.dTime = D.copy(timestep)

        return timestep, (self.dTime, self.dState)

    def get_error_estimate(self):
        if hasattr(self, "final_state") and self.adaptive:
            p1 = D.sum(self.final_state[0][self.tableau_idx_expand] * self.aux, axis=0)
            p2 = D.sum(self.final_state[1][self.tableau_idx_expand] * self.aux, axis=0)
            return p1 - p2
        else:
            return D.zeros_like(self.dState)

class ExplicitSymplecticIntegrator(TableauIntegrator):
    """
    A base class for all symplectic numerical integration methods.

    A ExplicitSymplecticIntegrator derived object corresponds to a
    numerical integrator tailored to a particular dynamical system 
    with an integration scheme defined by the sequence of drift-kick
    coefficients in tableau.
    
    An explicit symplectic integrator may be considered as a sequence of carefully
    picked drift and kick stages that build off the previous stage which is
    the implementation considered here. A masking array of indices indicates
    the drift and kick variables that are updated at each stage.
    
    In a system defined by a Hamiltonian of q and p (generalised position and
    generalised momentum respectively), the drift stages update q and the kick
    stages update p. For a conservative Hamiltonian, a symplectic method will
    minimise the drift in the Hamiltonian during the integration.
    
    Attributes
    ----------
    tableau : numpy array, shape (N, N+1)
        A numpy array with N stages and 3 entries per stage where the first column
        is the timestep fraction and the remaining columns are the drift/kick coefficients.
        
    symplectic : bool
        True if the method is symplectic.
    """

    symplectic = True

    # Class properties for accessing attributes of the class #
    @classmethod
    def is_adaptive(cls):
        return False

    @classmethod
    def is_explicit(cls):
        return True

    @classmethod
    def is_implicit(cls):
        return False

    @classmethod
    def is_fsal(cls):
        return False
    # ---- #

    # Instance properties that are cached #
    @property
    def adaptive(self):
        return False

    @adaptive.setter
    def adaptive(self, other):
        pass

    @property
    def fsal(self):
        return False

    @property
    def explicit(self):
        return True

    @property
    def implicit(self):
        return False

    @property
    def explicit_stages(self):
        return None

    @property
    def implicit_stages(self):
        return None
    # ---- #

    def __init__(self, sys_dim, dtype=None, staggered_mask=None, rtol=None, atol=None, device=None):
        super().__init__(sys_dim=sys_dim, dtype=dtype, rtol=rtol, atol=atol, device=device)
        if staggered_mask is None:
            staggered_mask = D.arange(sys_dim[0] // 2, sys_dim[0], dtype=D.int64)
            self.staggered_mask = D.zeros(sys_dim, dtype=D.bool)
            self.staggered_mask[staggered_mask] = 1
        else:
            self.staggered_mask = D.to_type(staggered_mask, D.bool)

        self.msk = self.staggered_mask
        self.nmsk = D.logical_not(self.staggered_mask)

        if D.backend() == 'torch':
            self.msk = self.msk.to(self.tableau)
            self.nmsk = self.nmsk.to(self.tableau)

    def __call__(self, rhs, initial_time, initial_state, constants, timestep):
        self.initial_state = D.copy(initial_state)
        self.initial_time = D.copy(initial_time)
        self.initial_rhs = None

        self.step(rhs=rhs, initial_time=initial_time, initial_state=initial_state,
                  constants=constants, timestep=timestep)

        if self.initial_rhs is None:
            self.initial_rhs = rhs(initial_time, initial_state, **constants)

        if self.final_rhs is None:
            self.final_rhs = rhs(initial_time + self.dTime, initial_state + self.dState, **constants)

        return timestep, (self.dTime, self.dState)

    def dense_output(self):
        return (self.initial_time + self.dTime,
                utilities.interpolation.CubicHermiteInterp(
                    self.initial_time,
                    self.initial_time + self.dTime,
                    self.initial_state,
                    self.initial_state + self.dState,
                    self.initial_rhs,
                    self.final_rhs
                ))

    def step(self, rhs, initial_time, initial_state, constants, timestep):
        msk = self.msk
        nmsk = self.nmsk

        current_time = D.copy(initial_time)
        self.dState *= 0.0

        for stage in range(D.shape(self.tableau)[0]):
            if stage == 0:
                self.initial_rhs = rhs(current_time, initial_state + self.dState, **constants)
                aux = timestep * self.initial_rhs
            else:
                aux = timestep * rhs(current_time, initial_state + self.dState, **constants)
            current_time = current_time + timestep * self.tableau[stage, 0]
            self.dState += aux * self.tableau[stage, 1] * msk + aux * self.tableau[stage, 2] * nmsk

        self.dTime = D.copy(timestep)

        return self.dTime, (self.dTime, self.dState)


def generate_richardson_integrator(basis_integrator):
    """
    A function for generating an integrator that uses local Richardson Extrapolation to find the change in state ΔY over a timestep h by estimating lim ΔY as h->0.
    
    Takes any integrator as input and returns a specialisation of the RichardsonExtrapolatedIntegrator class that uses basis_integrator as the underlying integration mechanism.
    
    Parameters
    ----------
    basis_integrator : A subclass of IntegratorTemplate or a class that implements the methods and attributes of IntegratorTemplate.

    Returns
    -------
    RichardsonExtrapolatedIntegrator
        returns the Richardson Extrapolated specialisation of basis_integrator
    """

    class RichardsonExtrapolatedIntegrator(RichardsonIntegratorTemplate):
        __alt_names__ = ("Local Richardson Extrapolation of {}".format(basis_integrator.__name__),)

        # Class properties for accessing attributes of the class #
        @classmethod
        def is_adaptive(cls):
            return True

        @classmethod
        def is_explicit(cls):
            return basis_integrator.is_explicit()

        @classmethod
        def is_implicit(cls):
            return basis_integrator.is_implicit()

        @classmethod
        def is_fsal(cls):
            return False

        # ---- #

        # Instance properties that are cached #
        @property
        def adaptive(self):
            return True

        @adaptive.setter
        def adaptive(self, other):
            self.__adaptive = bool(other)

        @property
        def fsal(self):
            return False

        @property
        def explicit(self):
            return self.basis_integrators[0].explicit

        @property
        def implicit(self):
            return self.basis_integrators[0].implicit

        @property
        def explicit_stages(self):
            return None

        @property
        def implicit_stages(self):
            return None
        # ---- #

        symplectic = basis_integrator.symplectic

        def __init__(self, sys_dim, richardson_iter=8, **kwargs):
            super().__init__()
            self.dim = sys_dim
            self.numel = 1
            for i in self.dim:
                self.numel *= int(i)
            self.rtol = kwargs.get("rtol") if kwargs.get("rtol", None) is not None else 32 * D.epsilon()
            self.atol = kwargs.get("atol") if kwargs.get("atol", None) is not None else 32 * D.epsilon()
            self.dtype = kwargs.get("dtype", )
            self.aux = D.zeros((richardson_iter, richardson_iter, *self.dim), dtype=self.dtype)
            self.dState = D.zeros(self.dim, dtype=self.dtype)
            self.dTime = D.zeros(tuple(), dtype=self.dtype)
            self.device = kwargs.get("device", None)
            self.initial_state = None
            self.initial_rhs = None
            self.initial_time = None
            self.final_rhs = None
            self.__adaptive = True
            self.__fsal = False
            self.tableau_idx_expand = tuple([slice(1, None, None)] + [None] * (self.aux.ndim - 1))
            self.richardson_iter = richardson_iter
            self.basis_integrators = [basis_integrator(sys_dim, **kwargs) for _ in range(self.richardson_iter)]
            for integrator in self.basis_integrators:
                integrator.adaptive = False
            self.basis_order = self.basis_integrators[0].order
            self.order = self.basis_order + richardson_iter // 2
            if 'staggered_mask' in kwargs:
                if kwargs['staggered_mask'] is None:
                    staggered_mask = D.arange(sys_dim[0] // 2, sys_dim[0], dtype=D.int64)
                    self.staggered_mask = D.zeros(sys_dim, dtype=D.bool)
                    self.staggered_mask[staggered_mask] = 1
                else:
                    self.staggered_mask = D.to_type(kwargs['staggered_mask'], D.bool)

            if self.dtype is not None:
                if D.backend() == 'torch':
                    self.aux = self.aux.to(self.dtype)
                else:
                    self.aux = self.aux.astype(self.dtype)

            if D.backend() == 'torch':
                self.aux = self.aux.to(self.device)

            self.__interpolants = None
            self.__interpolant_times = None
            self.solver_dict = dict(safety_factor=0.5 if self.implicit else 0.9, atol=self.atol, rtol=self.rtol)

        def dense_output(self):
            return self.__interpolant_times, self.__interpolants

        def check_converged(self, initial_state, diff, prev_error):
            err_estimate = D.max(D.abs(D.to_float(diff)))
            relerr = D.max(D.to_float(self.atol + self.rtol * D.abs(initial_state)))
            if prev_error is None or (err_estimate > relerr and err_estimate <= D.max(D.abs(D.to_float(prev_error)))):
                return diff, False
            else:
                return diff, True

        def subdiv_step(self, int_num, rhs, initial_time, initial_state, timestep, constants, num_intervals):
            dt_now, dstate_now = 0.0, 0.0
            dtstep = timestep / num_intervals
            self.__interpolants = []
            self.__interpolant_times = []
            for interval in range(num_intervals):
                dt, (dt_z, dy_z) = self.basis_integrators[int_num](rhs, initial_time + dt_now,
                                                                   initial_state + dstate_now,
                                                                   constants, dtstep)
                dt_now = dt_now + dt_z
                dstate_now = dstate_now + dy_z
                __interp_t, __interp = self.basis_integrators[int_num].dense_output()
                self.__interpolant_times.append(__interp_t)
                self.__interpolants.append(__interp)
            return dtstep, (dt_now, dstate_now)

        def adaptive_richardson(self, rhs, t, y, constants, timestep):
            dt0, (dt_z, dy_z) = self.subdiv_step(0, rhs, t, y, timestep, constants, 1)
            if dt_z < timestep:
                timestep = dt_z
            self.aux[0, 0] = dy_z
            prev_error = None
            m, n = 0, 0
            for m in range(1, self.richardson_iter):
                self.aux[m, 0] = self.subdiv_step(m, rhs, t, y, timestep, constants, 1 << m)[1][1]
                for n in range(1, m + 1):
                    self.aux[m, n] = self.aux[m, n - 1] + (self.aux[m, n - 1] - self.aux[m - 1, n - 1]) / (
                            (1 << n) - 1)
                self.order = self.basis_order + m + 1
                if m >= 3:
                    prev_error, t_conv = self.check_converged(self.aux[m, n],
                                                              self.aux[m - 1, m - 1] - self.aux[m, m],
                                                              prev_error)
                    if t_conv:
                        break

            return timestep, (timestep, self.aux[m - 1, n - 1]), self.aux[m - 1, m - 1] - self.aux[m, m]

        def __call__(self, rhs, initial_time, initial_state, constants, timestep):
            dt0, (dt_z, dy_z), diff = self.adaptive_richardson(rhs, initial_time, initial_state, constants,
                                                               timestep)

            self.dState = dy_z + 0.0
            self.dTime = D.copy(dt_z)

            self.solver_dict['diff'] = diff
            self.solver_dict['initial_state'] = initial_state
            self.solver_dict['initial_time'] = initial_time
            self.solver_dict['timestep'] = self.dTime
            self.solver_dict['dState'] = self.dState
            self.solver_dict['order'] = self.order
            new_timestep, redo_step = self.update_timestep()
            if self.symplectic:
                timestep = dt0
                next_timestep = D.copy(dt0)
                if (0.8 * new_timestep + 0.2 * timestep) < next_timestep:
                    while new_timestep < next_timestep:
                        next_timestep /= 2.0
                else:
                    while (0.8 * new_timestep + 0.2 * timestep) > 2 * next_timestep:
                        next_timestep *= 2.0
                    redo_step = False
            else:
                next_timestep = new_timestep

            if redo_step:
                timestep, (self.dTime, self.dState) = self(rhs, initial_time, initial_state, constants,
                                                           next_timestep)
            else:
                timestep = next_timestep

            return timestep, (self.dTime, self.dState)

    RichardsonExtrapolatedIntegrator.__name__ = RichardsonExtrapolatedIntegrator.__qualname__ = "RichardsonExtrapolated_{}_Integrator".format(
        basis_integrator.__name__)

    return RichardsonExtrapolatedIntegrator
