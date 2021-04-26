from .integrator_template import IntegratorTemplate, RichardsonIntegratorTemplate
from .. import backend as D
from .. import utilities
from .. import exception_types

__all__ = [
    'ExplicitRungeKuttaIntegrator',
    'ExplicitSymplecticIntegrator',
    'ImplicitRungeKuttaIntegrator',
    'generate_richardson_integrator'
]

class ExplicitIntegrator(IntegratorTemplate):
    __implicit__ = False
    
    def __init__(self, sys_dim, aux_shape, adaptive, dtype=None, rtol=None, atol=None, device=None):
        self.dim        = sys_dim
        self.numel      = 1
        for i in self.dim:
            self.numel *= int(i)
        self.rtol       = rtol if rtol is not None else 32 * D.epsilon()
        self.atol       = atol if atol is not None else 32 * D.epsilon()
        self.adaptive   = adaptive
        self.dtype      = dtype
        self.aux        = D.zeros((*aux_shape, *self.dim), dtype=self.dtype)
        self.dState     = D.zeros(self.dim, dtype=self.dtype)
        self.dTime      = D.zeros(tuple(), dtype=self.dtype)
        self.device     = device
        self.initial_state = None
        self.initial_rhs   = None
        self.initial_time  = None
        self.final_rhs     = None
    
class ImplicitIntegrator(IntegratorTemplate):
    __implicit__ = True
    
    def __init__(self, sys_dim, aux_shape, adaptive, dtype=None, rtol=None, atol=None, device=None):
        self.dim        = sys_dim
        self.numel      = 1
        for i in self.dim:
            self.numel *= int(i)
        self.rtol       = rtol if rtol is not None else 32 * D.epsilon()
        self.atol       = atol if atol is not None else 32 * D.epsilon()
        self.adaptive   = adaptive
        self.dtype      = dtype
        self.aux        = D.zeros((*aux_shape, *self.dim), dtype=self.dtype)
        self.dState     = D.zeros(self.dim, dtype=self.dtype)
        self.dTime      = D.zeros(tuple(), dtype=self.dtype)
        self.device     = device
        self.initial_state = None
        self.initial_rhs   = None
        self.initial_time  = None
        self.final_rhs     = None

class ExplicitRungeKuttaIntegrator(ExplicitIntegrator):
    """
    A base class for all explicit Runge-Kutta methods with a lower triangular Butcher Tableau.

    An ExplicitRungeKuttaIntegrator derived object corresponds to a
    numerical integrator tailored to a particular dynamical system 
    with an integration scheme defined by the Butcher tableau of the child
    class.
    
    A child class that defines two sets of coefficients for final_state
    is considered an adaptive method and uses the adaptive stepping 
    based on the local error estimate derived from the two sets of 
    final_state coefficients. Furthermore, local extrapolation is used.
    
    Attributes
    ----------
    tableau : numpy array, shape (N, N+1)
        A numpy array with N stages and N+1 entries per stage where the first column 
        is the timestep fraction and the remaining columns are the stage coefficients.
        
    final_state : numpy array, shape (k, N)
        A numpy array with N+1 coefficients defining the final stage coefficients.
        If k == 2, then the method is considered adaptive and the first row is
        the lower order method and the second row is the higher order method
        whose difference gives the local error of the numerical integration.
        
    __symplectic__ : bool
        True if the method is symplectic.
    """
    
    tableau = None
    final_state = None
    order = 1
    __symplectic__ = False

    def __init__(self, sys_dim, dtype=None, rtol=None, atol=None, device=None):
        super().__init__(sys_dim, (D.shape(self.tableau)[0],), D.shape(self.final_state)[0] == 2, dtype, rtol, atol, device)
        if dtype is None:
            self.tableau     = D.array(self.tableau)
            self.final_state = D.array(self.final_state)
        else:
            self.tableau     = D.to_type(self.tableau, dtype)
            self.final_state = D.to_type(self.final_state, dtype)
            
        self.__fsal = bool(D.all(self.tableau[-1, 1:] == self.final_state[0][1:]))
        self.fsal   = self.__fsal
        
        if D.backend() == 'torch':
            self.aux         = self.aux.to(self.device)
            self.dState      = self.dState.to(self.device)
            self.dTime       = self.dTime.to(self.device)
            self.tableau     = self.tableau.to(self.device)
            self.final_state = self.final_state.to(self.device)
            
    def forward(self, rhs, initial_time, initial_state, constants, timestep):
        if self.tableau is None:
            raise NotImplementedError("In order to use the fixed step integrator, subclass this class and populate the butcher tableau")
        else:
            tableau_idx_expand = tuple([slice(1, None, None)] + [None] * (self.aux.ndim - 1))
            self.initial_state = D.copy(initial_state)
            self.initial_time  = D.copy(initial_time)
            self.initial_rhs   = None
            if not self.__fsal:
                self.final_rhs = None

            for stage in range(D.shape(self.aux)[0]):
                current_state   = initial_state    + timestep * D.sum(self.tableau[stage][tableau_idx_expand] * self.aux, axis=0)
                if self.__fsal and self.final_rhs is not None and stage == 0:
                    self.aux[stage] = self.final_rhs
                else:
                    self.aux[stage] = rhs(initial_time + self.tableau[stage, 0]*timestep, current_state, **constants)
                
                if stage == 0 and D.sum(self.tableau[0]) == 0.0:
                    self.initial_rhs = self.aux[0]
                elif self.__fsal and stage == D.shape(self.aux)[0] - 1:
                    self.final_rhs   = self.aux[stage]
            
            if self.__fsal:
                self.dState = self.aux[-1] / timestep
            else:
                self.dState = timestep * D.sum(self.final_state[0][tableau_idx_expand] * self.aux, axis=0)
                
            self.dTime  = D.copy(timestep)
                
            if self.initial_rhs is None:
                self.initial_rhs = rhs(initial_time, initial_state, **constants)
                
            if self.final_rhs is None:
                self.final_rhs = rhs(initial_time + self.dTime, initial_state + self.dState, **constants)
            
            if self.adaptive:
                diff = timestep * self.get_error_estimate(tableau_idx_expand)
                timestep, redo_step = self.update_timestep(initial_state, self.dState, diff, initial_time, timestep)
                if redo_step:
                    timestep, (self.dTime, self.dState) = self(rhs, initial_time, initial_state, constants, timestep)
            
            return timestep, (self.dTime, self.dState)
        
    def get_error_estimate(self, tableau_idx_expand):
        return D.sum(self.final_state[1][tableau_idx_expand] * self.aux, axis=0) - D.sum(self.final_state[0][tableau_idx_expand] * self.aux, axis=0)

    __call__ = forward
    
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

class ExplicitSymplecticIntegrator(ExplicitIntegrator):
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
        A numpy array with N stages and N+1 entries per stage where the first column 
        is the timestep fraction and the remaining columns are the stage coefficients.
        
    __symplectic__ : bool
        True if the method is symplectic.
    """
    
    tableau = None
    __symplectic__ = True

    def __init__(self, sys_dim, dtype=None, staggered_mask=None, rtol=None, atol=None, device=None):
        super().__init__(sys_dim, (D.shape(self.tableau)[0],), False, dtype, rtol, atol, device)
        if staggered_mask is None:
            staggered_mask      = D.arange(sys_dim[0]//2, sys_dim[0], dtype=D.int64)
            self.staggered_mask = D.zeros(sys_dim, dtype=D.bool)
            self.staggered_mask[staggered_mask] = 1
        else:
            self.staggered_mask = D.to_type(staggered_mask, D.bool)
            
        if dtype is None:
            self.tableau     = D.array(self.tableau)
        else:
            self.tableau     = D.to_type(self.tableau, dtype)
            
        self.msk  = self.staggered_mask
        self.nmsk = D.logical_not(self.staggered_mask)
        
        if D.backend() == 'torch':
            self.dState  = self.dState.to(self.device)
            self.dTime   = self.dTime.to(self.device)
            self.tableau = self.tableau.to(self.device)
            self.msk     = self.msk.to(self.tableau)
            self.nmsk    = self.nmsk.to(self.tableau)

    def forward(self, rhs, initial_time, initial_state, constants, timestep):
        if self.tableau is None:
            raise NotImplementedError("In order to use the fixed step integrator, subclass this class and populate the butcher tableau")
        else:
            msk  = self.msk
            nmsk = self.nmsk

            current_time  = D.copy(initial_time)
            self.dState  *= 0.0
            self.initial_state = D.copy(initial_state)
            self.initial_time  = D.copy(initial_time)
            self.initial_rhs   = None
            self.final_rhs     = None

            for stage in range(D.shape(self.tableau)[0]):
                aux          = timestep * rhs(current_time, initial_state + self.dState, **constants)
                current_time = current_time + timestep * self.tableau[stage, 0]
                self.dState += aux * self.tableau[stage, 1] * msk + aux * self.tableau[stage, 2] * nmsk
                
            self.dTime = D.copy(timestep)
            self.initial_rhs = rhs(current_time, initial_state, **constants)
            self.final_rhs   = rhs(initial_time + self.dTime, initial_state + self.dState, **constants)
            
            return self.dTime, (self.dTime, self.dState)
        
    __call__ = forward
    
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
    
class ImplicitRungeKuttaIntegrator(ImplicitIntegrator):
    """
    A base class for all implicit Runge-Kutta methods with arbitrary Butcher Tableau.

    An ImplicitRungeKuttaIntegrator derived object corresponds to a
    numerical integrator tailored to a particular dynamical system 
    with an integration scheme defined by the Butcher tableau of the child
    class.
    
    A child class that defines two sets of coefficients for final_state
    is considered an adaptive method and uses the adaptive stepping 
    based on the local error estimate derived from the two sets of 
    final_state coefficients. Furthermore, local extrapolation is used.
    
    Attributes
    ----------
    tableau : numpy array, shape (N, N+1)
        A numpy array with N stages and N+1 entries per stage where the first column 
        is the timestep fraction and the remaining columns are the stage coefficients.
        
    final_state : numpy array, shape (k, N)
        A numpy array with N+1 coefficients defining the final stage coefficients.
        If k == 2, then the method is considered adaptive and the first row is
        the lower order method and the second row is the higher order method
        whose difference gives the local error of the numerical integration.
        
    __symplectic__ : bool
        True if the method is symplectic.
    """
    
    tableau = None
    final_state = None
    order = 1
    __symplectic__ = False

    def __init__(self, sys_dim, dtype=None, rtol=None, atol=None, device=None):
        super().__init__(sys_dim, (D.shape(self.tableau)[0],), D.shape(self.final_state)[0] == 2, dtype, rtol, atol, device)
        if dtype is None:
            self.tableau     = D.array(self.tableau)
            self.final_state = D.array(self.final_state)
        else:
            self.tableau     = D.to_type(self.tableau, dtype)
            self.final_state = D.to_type(self.final_state, dtype)
        
        if D.backend() == 'torch':
            self.dState      = self.dState.to(self.device)
            self.dTime       = self.dTime.to(self.device)
            self.aux         = self.aux.to(self.device)
            self.tableau     = self.tableau.to(self.device)
            self.final_state = self.final_state.to(self.device)
            
    def forward(self, rhs, initial_time, initial_state, constants, timestep):
        if self.tableau is None:
            raise NotImplementedError("In order to use the fixed step integrator, subclass this class and populate the butcher tableau")
        else:
            tableau_idx_expand = tuple([slice(1, None, None)] + [None] * (self.aux.ndim - 1))

        self.initial_state = D.copy(initial_state)
        self.initial_time  = D.copy(initial_time)
        self.initial_rhs   = None
        self.final_rhs     = None
        aux_shape = self.aux.shape
        
        def nfun(next_state):
            nonlocal initial_time, initial_state, timestep, aux_shape
            __aux_states = D.reshape(next_state, aux_shape)
            __rhs_states = D.stack([
                rhs(initial_time + tbl[0] * timestep, initial_state + timestep * D.sum(tbl[tableau_idx_expand] * __aux_states, axis=0), **constants) for tbl in self.tableau
            ])
            __states =  D.reshape(__aux_states - __rhs_states, (-1,))
            return __states
        
        def __nfun_jac(next_state):
            nonlocal self, initial_time, initial_state, timestep, aux_shape
            __aux_states = D.reshape(next_state, aux_shape)
            __step       = self.numel
            if D.backend() == 'torch':
                __jac        = D.eye(self.tableau.shape[0]*__step, device=__aux_states.device, dtype=__aux_states.dtype)
            else:
                __jac        = D.eye(self.tableau.shape[0]*__step)
            __prev_idx   = -1
            __rhs_jac = D.stack([
                rhs.jac(initial_time + tbl[0] * timestep, initial_state + timestep * D.sum(tbl[tableau_idx_expand] * __aux_states, axis=0), **constants) for tbl in self.tableau
            ])
            for idx in range(0,__jac.shape[0],__step):
                for jdx in range(0,__jac.shape[1],__step):
                    __jac[idx:idx+__step, jdx:jdx+__step] -= timestep * self.tableau[idx//__step, 1+jdx//__step]*__rhs_jac[idx//__step].reshape(__step, __step)
            if __jac.shape[0] == 1 and __jac.shape[1] == 1:
                __jac = D.reshape(__jac, tuple())
            return __jac
            
        initial_guess = D.copy(self.aux)
        
        midpoint_guess = rhs(initial_time, initial_state, **constants)
        self.initial_rhs   = midpoint_guess
        midpoint_guess = 0.5 * timestep * (midpoint_guess + rhs(initial_time + 0.5 * timestep, initial_state + 0.5 * timestep * midpoint_guess, **constants))
        
        initial_guess = D.to_float(initial_guess + (0.5*midpoint_guess + 0.5*self.dState)[None])
        
        if rhs.jac_is_wrapped_rhs and D.backend() == 'torch':
            nfun_jac = None
            initial_guess.requires_grad = True
        else:
            nfun_jac = __nfun_jac
            
        aux_root, (success, num_iter, prec) = utilities.optimizer.newtonraphson(nfun, initial_guess, jac=nfun_jac, verbose=False, tol=None, maxiter=70)
        if not success and prec > D.max(D.abs(self.atol + self.rtol * D.max(D.abs(D.to_float(initial_state))))):
            raise exception_types.FailedToMeetTolerances("Step size too large, cannot solve system to the tolerances required: achieved = {}, desired = {}, iter = {}".format(prec, 32*D.epsilon(), num_iter))
            
        self.aux    = D.reshape(aux_root, aux_shape)
#         if initial_state.dtype == object:
#             new_aux = D.stack([
#                     rhs(initial_time + tbl[0] * timestep, initial_state + timestep * D.sum(tbl[tableau_idx_expand] * self.aux, axis=0), **constants) for tbl in self.tableau
#                 ])
#             self.aux = new_aux
        self.dState = timestep * D.sum(self.final_state[0][tableau_idx_expand] * self.aux, axis=0)
        self.dTime  = D.copy(timestep)
        
        if self.final_rhs is None:
            self.final_rhs = rhs(initial_time + self.dTime, initial_state + self.dState, **constants)
        
        if self.adaptive:
            diff =  timestep * self.get_error_estimate(tableau_idx_expand)
            timestep, redo_step = self.update_timestep(initial_state, self.dState, diff, initial_time, timestep)
            if redo_step:
                timestep, (self.dTime, self.dState) = self(rhs, initial_time, initial_state, constants, timestep)

        return timestep, (self.dTime, self.dState)
        
    def update_timestep(self, initial_state, dState, diff, initial_time, timestep, tol=0.8):
        err_estimate = D.max(D.abs(D.to_float(diff)))
        relerr = D.max(D.to_float(self.atol + self.rtol * D.abs(initial_state) + self.rtol * D.abs(dState / timestep)))
        corr = 1.0
#         rel_niter = (self.niter0 / self.niter1)**(1.0 / self.order)
#         if rel_niter != 0:
#             corr = corr * rel_niter
#         else:
        if err_estimate != 0:
            corr = corr * tol * (relerr / err_estimate)**(1.0 / self.order)
        if corr != 0:
            timestep = corr * timestep
        if err_estimate > relerr:
            return timestep, True
        else:
            return timestep, False
        
    def get_error_estimate(self, tableau_idx_expand):
        return D.sum(self.final_state[1][tableau_idx_expand]* self.aux, axis=0) - D.sum(self.final_state[0][tableau_idx_expand]* self.aux, axis=0)

    __call__ = forward
    
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
    if issubclass(basis_integrator, ExplicitIntegrator):
        first_base = ExplicitIntegrator
    elif issubclass(basis_integrator, ImplicitIntegrator):
        first_base = ImplicitIntegrator
    else:
        first_base = IntegratorTemplate
    class RichardsonExtrapolatedIntegrator(first_base, RichardsonIntegratorTemplate):
        __alt_names__ = ("Local Richardson Extrapolation of {}".format(basis_integrator.__name__),)
        __symplectic__ = basis_integrator.__symplectic__
        __adaptive__   = True
        __implicit__   = basis_integrator.__implicit__

        def __init__(self, sys_dim, richardson_iter=8, **kwargs):
            super().__init__(sys_dim, (richardson_iter, richardson_iter), True, kwargs.get('dtype'), kwargs.get('rtol', 1e-3), kwargs.get('atol', 1e-3), kwargs.get('device', None))
            self.richardson_iter      = richardson_iter
            self.basis_integrators = [basis_integrator(sys_dim, **kwargs) for _ in range(self.richardson_iter)]
            for integrator in self.basis_integrators:
                integrator.adaptive = False
            self.basis_order       = basis_integrator.order
            self.order             = self.basis_order + 3
            if 'staggered_mask' in kwargs:
                if kwargs['staggered_mask'] is None:
                    staggered_mask      = D.arange(sys_dim[0]//2, sys_dim[0], dtype=D.int64)
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
        
    RichardsonExtrapolatedIntegrator.__name__ = RichardsonExtrapolatedIntegrator.__qualname__ = "RichardsonExtrapolated_{}_Integrator".format(basis_integrator.__name__)
    
    return RichardsonExtrapolatedIntegrator
