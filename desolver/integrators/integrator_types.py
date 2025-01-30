from desolver.integrators.integrator_template import IntegratorTemplate, RichardsonIntegratorTemplate
from desolver import backend as D
from desolver import utilities
from desolver import exception_types
from desolver.integrators import utilities as integrator_utilities
from desolver.integrators import components
import math
import abc

__all__ = [
    'RungeKuttaIntegrator',
    'ExplicitSymplecticIntegrator',
    'generate_richardson_integrator'
]


class TableauIntegrator(IntegratorTemplate, abc.ABC):
    tableau_intermediate = None # Stores the c and a-coefficients of a tableau-based integrator integrator
    __order__ = None

    def __init__(self, sys_dim, dtype, rtol=None, atol=None, device=None):
        super().__init__()
        self.dim = sys_dim
        self.numel = 1
        for i in self.dim:
            self.numel *= int(i)
        self.rtol = rtol if rtol is not None else 32 * D.epsilon(dtype)
        self.atol = atol if atol is not None else 32 * D.epsilon(dtype)
        self.dtype = dtype
        self.device = device
        self.array_constructor_kwargs = dict(dtype=self.dtype)
        # THIS IS WHERE WE SPECIALISE TO DIFFERENT BACKENDS
        if D.backend_like_dtype(self.dtype) == 'torch':
            self.array_constructor_kwargs['device'] = self.device
        
        self.dState = D.ar_numpy.zeros(self.dim, **self.array_constructor_kwargs)
        self.dTime = D.ar_numpy.zeros(tuple(), **self.array_constructor_kwargs)
        self.tableau_intermediate = D.ar_numpy.asarray(self.__class__.tableau_intermediate, **self.array_constructor_kwargs)
        
        self.initial_state = None
        self.initial_rhs = None
        self.initial_time = None
        
        self.final_state = None
        self.final_rhs = None
        self.final_time = None
        
        self._explicit = None
        self._adaptive = False
        self._fsal = False
        self._adaptivity_enabled = self._adaptive
        self._explicit_stages = None
        self._implicit_stages = None

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
    def is_implicit(self):
        return not self._explicit
    
    @property
    def is_explicit(self):
        return self._explicit
    
    @property
    def is_fsal(self):
        return self._fsal
    
    @property
    def is_adaptive(self):
        return self._adaptive and not self._adaptivity_enabled
    
    @is_adaptive.setter
    def is_adaptive(self, adaptivity):
        self._adaptivity_enabled = adaptivity

    @property
    def stages(self):
        return self.tableau_intermediate.shape[0]

    @property
    def explicit_stages(self):
        return self._explicit_stages

    @property
    def implicit_stages(self):
        return self._implicit_stages
    # ---- #
    
    # Required Functions for the Integrator Class
    @abc.abstractmethod
    def step(self, rhs, initial_time, initial_state, constants, timestep):
        pass
    # ---- #
    
    # Optional Functions for the Integrator Class #
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
    # ---- #


class RungeKuttaIntegrator(TableauIntegrator, abc.ABC):
    tableau_final = None # Stores the b/bh-coefficients of a Runge-Kutta integrator

    def __init__(self, sys_dim, dtype, rtol=None, atol=None, device=None):
        super().__init__(sys_dim=sys_dim, dtype=dtype, rtol=rtol, atol=atol, device=device)
        self.tableau_final = D.ar_numpy.asarray(self.__class__.tableau_final, **self.array_constructor_kwargs)

        # Initialise storage for intermediate stages #
        self.stage_values = D.ar_numpy.zeros((*self.dim, self.stages), **self.array_constructor_kwargs)
        # ---- #
        
        self._adaptive = self.tableau_final.shape[0] == 2
        self._fsal = bool(D.ar_numpy.all(self.tableau_intermediate[-1, 1:] == self.tableau_final[0, 1:]))
        self._explicit = all([(self.tableau_intermediate[col, col + 1:] == 0.0).all() for col in range(self.tableau_intermediate.shape[0])])
        self._explicit_stages = [col for col in range(self.stages) if D.ar_numpy.all(self.tableau_intermediate[col, col + 1:] == 0.0)]
        self._implicit_stages = [col for col in range(self.stages) if D.ar_numpy.any(self.tableau_intermediate[col, col + 1:] != 0.0)]

        self.solver_dict_preserved = dict(safety_factor=0.8, order=self.order, atol=self.atol, rtol=self.rtol, redo_count=0)
        self.solver_dict = dict()
        self.solver_dict.update(self.solver_dict_preserved)
        self.solver_dict.update(dict(
            initial_state=self.stage_values[...,0],
            diff=0.0,
            timestep=1.0,
            dState=self.stage_values[...,0]
        ))
        if self.is_implicit:
            self.solver_dict.update(dict(
                tau0=0, tau1=0, niter0=0, niter1=0
            ))
            self.adaptation_fn = integrator_utilities.implicit_aware_update_timestep

    def __call__(self, rhs, initial_time, initial_state, constants, timestep):
        self.solver_dict = self.solver_dict_preserved
        self.initial_state = D.ar_numpy.copy(initial_state)
        self.initial_time = D.ar_numpy.copy(initial_time)
        self.initial_rhs = None
        
        if self.is_fsal and self.final_rhs is not None:
            self.aux[0] = self.final_rhs
            self.initial_rhs = self.final_rhs
        self.final_rhs = None

        self.step(rhs=rhs, initial_time=initial_time, initial_state=initial_state,
                  constants=constants, timestep=timestep)

        if self.is_adaptive or self.is_implicit:
            self.solver_dict['redo_count'] = 0
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
                    self.solver_dict['redo_count'] += 1
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
        

    def algebraic_system(self, next_state, rhs, initial_time, initial_state, timestep, constants):
        __aux_states = D.ar_numpy.reshape(next_state, self.aux.shape)
        __rhs_states = D.ar_numpy.stack([
            rhs(initial_time + tbl[0] * timestep,
                initial_state + timestep * D.ar_numpy.sum(tbl[1:] * __aux_states, axis=-1), **constants)
            for tbl in self.tableau_intermediate
        ])
        __states = D.ar_numpy.reshape(__aux_states - __rhs_states, (-1,))
        return __states

    def algebraic_system_jacobian(self, next_state, rhs, initial_time, initial_state, timestep, constants):
        __aux_states = D.ar_numpy.reshape(next_state, (self.dim, self.stages))
        __step = self.numel
        __jac = D.ar_numpy.eye(self.tableau.shape[0] * __step, **self.array_constructor_kwargs)
        __rhs_jac = D.ar_numpy.stack([
            rhs.jac(initial_time + tbl[0] * timestep,
                    initial_state + timestep * D.ar_numpy.sum(tbl[1:] * __aux_states, axis=-1),
                    **constants)
            for tbl in self.tableau_intermediate
        ])
        for idx in range(0, __jac.shape[0], __step):
            for jdx in range(0, __jac.shape[1], __step):
                __jac[idx:idx + __step, jdx:jdx + __step] -= timestep * self.tableau[
                    idx // __step, 1 + jdx // __step] * __rhs_jac[idx // __step].reshape(__step, __step)
        if __jac.shape[0] == 1 and __jac.shape[1] == 1:
            __jac = D.ar_numpy.reshape(__jac, tuple())
        return __jac

    def step(self, rhs, initial_time, initial_state, constants, timestep):
        # Initial guess from assuming method is explicit #
        _, intermediate_dstate, intermediate_rhs = components.rk_methods.compute_step(
            rhs,
            initial_time,
            initial_state,
            timestep,
            self.stage_values,
            self.stage_values,
            self.tableau_intermediate,
            constants
        )

        if D.ar_numpy.sum(self.stage_values[...,0]) == 0.0:
            self.initial_rhs = self.stage_values[...,0]
        else:
            self.initial_rhs = None

        if self.is_implicit:
            raise TypeError("Implicit integrators are currently unsupported!")
            initial_guess = self.stage_values
            
            nfun_jac = None
            if D.autoray.backend_like(self.stage_values) == 'torch':
                nfun_jac = self.algebraic_system_jacobian

            desired_tol = D.ar_numpy.max(D.ar_numpy.abs(self.atol * 1e-1 + D.ar_numpy.max(D.ar_numpy.abs(D.ar_numpy.to_numpy(self.rtol * 1e-1 * initial_state)))))
            aux_root, (success, num_iter, _, _, prec) = \
                utilities.optimizer.nonlinear_roots(
                    self.algebraic_system, initial_guess,
                    jac=nfun_jac, verbose=False,
                    tol=desired_tol, maxiter=30,
                    additional_args=(rhs, initial_time, initial_state, timestep, constants))

            if not success and prec > desired_tol:
                raise exception_types.FailedToMeetTolerances(
                    "Step size too large, cannot solve system to the "
                    "tolerances required: achieved = {}, desired = {}, iter = {}".format(prec, desired_tol, num_iter))
            self.solver_dict.update(dict(
                tau0=self.solver_dict['tau1'], tau1=timestep,
                niter0=self.solver_dict['niter0'], niter1=num_iter
            ))

            self.aux = D.ar_numpy.reshape(aux_root, self.aux.shape)

        if self.is_fsal and self.is_explicit:
            self.dState = intermediate_dstate
            self.final_rhs = intermediate_rhs
        else:
            self.dState = timestep * D.ar_numpy.sum(self.stage_values * self.tableau_final[0, 1:], axis=-1)
        self.dTime = D.ar_numpy.copy(timestep)

        return timestep, (self.dTime, self.dState)

    def get_error_estimate(self):
        if self.tableau_final.shape[0] == 2 and self.is_adaptive:
            return D.ar_numpy.sum((self.tableau_final[0, 1:] - self.tableau_final[1, 1:]) * self.stage_values, axis=-1)
        else:
            return D.ar_numpy.zeros_like(self.dState)


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

    def __init__(self, sys_dim, dtype=None, staggered_mask=None, rtol=None, atol=None, device=None):
        super().__init__(sys_dim=sys_dim, dtype=dtype, rtol=rtol, atol=atol, device=device)
        if staggered_mask is None:
            staggered_mask = D.ar_numpy.arange(sys_dim[0] // 2, sys_dim[0], dtype=D.autoray.to_backend_dtype('int64', like=self.tableau_intermediate))
            self.staggered_mask = D.ar_numpy.zeros(sys_dim, dtype=D.autoray.to_backend_dtype('bool', like=self.tableau_intermediate))
            self.staggered_mask[staggered_mask] = 1
        else:
            self.staggered_mask = D.astype(staggered_mask, D.autoray.to_backend_dtype('bool', like=self.tableau_intermediate))

        self.kick_mask = D.ar_numpy.asarray(self.staggered_mask, **self.array_constructor_kwargs)
        self.drift_mask = 1.0 - self.kick_mask
        
        self._adaptive = False
        self._fsal = False
        self._explicit = True
        self._explicit_stages = self.stages
        self._implicit_stages = None

    def __call__(self, rhs, initial_time, initial_state, constants, timestep):
        self.initial_state = D.ar_numpy.copy(initial_state)
        self.initial_time = D.ar_numpy.copy(initial_time)
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
        current_time = D.ar_numpy.copy(initial_time)
        self.dState *= 0.0

        for stage in range(D.ar_numpy.shape(self.tableau_intermediate)[0]):
            if stage == 0:
                self.initial_rhs = rhs(current_time, initial_state + self.dState, **constants)
                aux = timestep * self.initial_rhs
            else:
                aux = timestep * rhs(current_time, initial_state + self.dState, **constants)
            current_time = current_time + timestep * self.tableau_intermediate[stage, 0]
            self.dState += aux * (self.tableau_intermediate[stage, 1] * self.drift_mask + self.tableau_intermediate[stage, 2] * self.kick_mask)

        self.dTime = D.ar_numpy.copy(timestep)

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
    raise TypeError("Richardson Extrapolated Integrators are currently unsupported!")
    class RichardsonExtrapolatedIntegrator(RichardsonIntegratorTemplate):
        __alt_names__ = ("Local Richardson Extrapolation of {}".format(basis_integrator.__name__),)

        symplectic = basis_integrator.is_symplectic

        def __init__(self, sys_dim, richardson_iter=8, **kwargs):
            super().__init__()
            self.dim = sys_dim
            self.numel = 1
            for i in self.dim:
                self.numel *= int(i)
            self.rtol = kwargs.get("rtol") if kwargs.get("rtol", None) is not None else 32 * D.epsilon()
            self.atol = kwargs.get("atol") if kwargs.get("atol", None) is not None else 32 * D.epsilon()
            self.dtype = kwargs.get("dtype")
            self.device = kwargs.get("device", None)
            self.array_constructor_args = dict(dtype=self.dtype)
            if self.device is not None:
                self.array_constructor_args['device'] = self.device
            self.aux = D.ar_numpy.zeros((richardson_iter, richardson_iter, *self.dim), **self.array_constructor_args)
            self.dState = D.ar_numpy.zeros(self.dim, **self.array_constructor_args)
            self.dTime = D.ar_numpy.zeros(tuple(), **self.array_constructor_args)
            self.initial_state = None
            self.initial_rhs = None
            self.initial_time = None
            self.final_rhs = None
            self.tableau_idx_expand = tuple([slice(1, None, None)] + [None] * (self.aux.ndim - 1))
            self.richardson_iter = richardson_iter
            self.basis_integrators = [basis_integrator(sys_dim, **kwargs) for _ in range(self.richardson_iter)]
            for integrator in self.basis_integrators:
                integrator.is_adaptive = False
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
            
            self._adaptive = True
            self._fsal = False
            self._explicit = self.basis_integrators[0].is_explicit

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
