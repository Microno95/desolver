from .integrator_template import IntegratorTemplate, RichardsonIntegratorTemplate
from .. import backend as D

__all__ = [
    'ExplicitRungeKuttaIntegrator',
    'ExplicitSymplecticIntegrator',
    'generate_richardson_integrator'
]

class ExplicitIntegrator(IntegratorTemplate):
    def __init__(self, sys_dim, aux_shape, adaptive, dtype=None, rtol=None, atol=None):
        self.dim        = sys_dim
        self.rtol       = rtol
        self.atol       = atol
        self.adaptive   = adaptive
        self.dtype      = dtype
        self.aux        = D.zeros(aux_shape + self.dim, dtype=self.dtype)
        
    def dense_output(self, rhs, initial_time, initial_state):
        return CubicHermiteInterp(
            initial_time, 
            initial_time + self.dTime, 
            initial_state, 
            initial_state + self.dState,
            rhs(initial_time, initial_state),
            rhs(initial_time + self.dTime, initial_state + self.dState)
        )

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
        super().__init__(sys_dim, (D.shape(self.tableau)[0],), D.shape(self.final_state)[0] == 2, dtype, rtol, atol)
        if dtype is None:
            self.tableau     = D.array(self.tableau)
            self.final_state = D.array(self.final_state)
        else:
            self.tableau     = D.to_type(self.tableau, dtype)
            self.final_state = D.to_type(self.final_state, dtype)
        
        if dtype is not None:
            if D.backend() == 'torch':
                self.aux = self.aux.to(dtype)
            else:
                self.aux = self.aux.astype(dtype)
        
        if D.backend() == 'torch':
            self.aux         = self.aux.to(device)
            self.tableau     = self.tableau.to(device)
            self.final_state = self.final_state.to(device)
            
    def forward(self, rhs, initial_time, initial_state, constants, timestep):
        if self.tableau is None:
            raise NotImplementedError("In order to use the fixed step integrator, subclass this class and populate the butcher tableau")
        else:
            tableau_idx_expand = tuple([slice(1, None, None)] + [None] * (self.aux.ndim - 1))

            for stage in range(D.shape(self.aux)[0]):
                current_state   = initial_state    + D.sum(self.tableau[stage][tableau_idx_expand] * self.aux, axis=0)
                self.aux[stage] = rhs(initial_time + self.tableau[stage, 0]*timestep, current_state, **constants) * timestep
                
                           
            self.dState = D.sum(self.final_state[0][tableau_idx_expand] * self.aux, axis=0)
            self.dTime  = D.copy(timestep)
            
            if self.adaptive:
                diff = self.get_error_estimate(self.dState, self.dTime, self.aux, tableau_idx_expand)
                timestep, redo_step = self.update_timestep(initial_state, self.dState, diff, initial_time, timestep)
                if redo_step:
                    timestep, (self.dTime, self.dState) = self(rhs, initial_time, initial_state, constants, timestep)
            
            return timestep, (self.dTime, self.dState)
        
    def get_error_estimate(self, dState, dTime, aux, tableau_idx_expand):
        return dState - D.sum(self.final_state[1][tableau_idx_expand] * aux, axis=0)

    __call__ = forward

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
        super().__init__(sys_dim, (D.shape(self.tableau)[0],), False, dtype, rtol, atol)
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
            self.tableau     = self.tableau.to(device)
            self.msk  = self.msk.to(self.tableau)
            self.nmsk = self.nmsk.to(self.tableau)

    def forward(self, rhs, initial_time, initial_state, constants, timestep):
        if self.tableau is None:
            raise NotImplementedError("In order to use the fixed step integrator, subclass this class and populate the butcher tableau")
        else:
            msk  = self.msk
            nmsk = self.nmsk

            current_time  = D.copy(initial_time)
            current_state = D.copy(initial_state)
            self.dState   = D.zeros_like(current_state)

            for stage in range(D.shape(self.tableau)[0]):
                aux          = rhs(current_time, initial_state + self.dState, **constants) * timestep
                current_time = current_time + timestep * self.tableau[stage, 0]
                self.dState += aux * self.tableau[stage, 1] * msk + aux * self.tableau[stage, 2] * nmsk
                
            self.dTime = D.copy(timestep)
            
            return self.dTime, (self.dTime, self.dState)
        
    __call__ = forward
    
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
    else:
        first_base = IntegratorTemplate
    class RichardsonExtrapolatedIntegrator(first_base, RichardsonIntegratorTemplate):
        __alt_names__ = ("Local Richardson Extrapolation of {}".format(basis_integrator.__name__),)
        __symplectic__ = basis_integrator.__symplectic__
        __adaptive__   = True

        def __init__(self, sys_dim, richardson_iter=8, **kwargs):
            super().__init__(sys_dim, (richardson_iter, richardson_iter), True, kwargs.get('dtype'), kwargs.get('rtol', 1e-3), kwargs.get('atol', 1e-3))
            self.richardson_iter      = richardson_iter
            self.basis_integrators = [basis_integrator(sys_dim, **kwargs) for _ in range(self.richardson_iter)]
            for integrator in self.basis_integrators:
                integrator.adaptive = False
            self.basis_order       = basis_integrator.order
            self.order             = self.basis_order + 3
            self.device            = kwargs.get('device')
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
        
    RichardsonExtrapolatedIntegrator.__name__ = RichardsonExtrapolatedIntegrator.__qualname__ = "RichardsonExtrapolated_{}_Integrator".format(basis_integrator.__name__)
    
    return RichardsonExtrapolatedIntegrator
