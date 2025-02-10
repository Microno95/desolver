import collections

from tqdm.auto import tqdm

from desolver import backend as D
from desolver import integrators as integrators
from desolver import exception_types as etypes
from desolver import utilities as deutil

import numpy as np

CubicHermiteInterp = deutil.interpolation.CubicHermiteInterp
root_finder = deutil.optimizer.brentsrootvec
root_polisher = deutil.optimizer.newtontrustregion

__all__ = [
    'DiffRHS',
    'rhs_prettifier',
    'OdeSystem'
]

StateTuple = collections.namedtuple('StateTuple', ['t', 'y', 'event'])


##### Code adapted from https://github.com/scipy/scipy/blob/v1.3.2/scipy/integrate/_ivp/ivp.py#L28 #####
def prepare_events(events, backend_like):
    """Standardize event functions and extract is_terminal and direction."""
    if callable(events):
        events = (events,)

    if events is not None:
        is_terminal = D.ar_numpy.zeros(len(events), dtype=D.autoray.to_backend_dtype("bool", like=backend_like), like=backend_like)
        direction = D.ar_numpy.zeros(len(events), dtype=D.autoray.to_backend_dtype("int64", like=backend_like), like=backend_like)
        last_occurrence = D.ar_numpy.zeros(len(events), dtype=D.autoray.to_backend_dtype("int64", like=backend_like), like=backend_like) - 1
        requires_dstate = D.ar_numpy.zeros(len(events), dtype=D.autoray.to_backend_dtype("bool", like=backend_like), like=backend_like)
        if D.autoray.infer_backend(backend_like) == 'torch':
            is_terminal = is_terminal.to(backend_like.device)
            direction = direction.to(backend_like.device)
            last_occurrence = last_occurrence.to(backend_like.device)
            requires_dstate = requires_dstate.to(backend_like.device)
        for i, event in enumerate(events):
            if hasattr(event, "is_terminal"):
                is_terminal[i] = bool(event.is_terminal)
            if hasattr(event, "direction"):
                direction[i] = event.direction
            if hasattr(event, "requires_dstate"):
                requires_dstate[i] = event.requires_dstate
    else:
        is_terminal = None
        direction = None
        last_occurrence = None
        requires_dstate = None

    return events, is_terminal, direction, last_occurrence, requires_dstate
##### ---- #####

def handle_events(sol_tuple, events, consts, direction, is_terminal, attributes):
    """Helper function to handle events.
    Parameters
    ----------
    sol : DenseOutput
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    events : list of callables
        List of Event functions
    consts : dict
        Dictionary of system constants
    direction : array-type, shape (n_events,)
        Direction of event to be detected
    is_terminal : array-type, shape (n_events,)
        Which events are terminal.
    Returns
    -------
    active_events : array-type
        Indices of events which take zero between `t_prev` and `t_next` and before
        a possible termination.
    roots : array-type
        Values of t at which events occurred sorted in time according to the direction
        of `t_prev` to `t_next`.
    terminate : bool
        Whether a terminal event occurred.
    """
    sol, t_prev, t_next = sol_tuple
    requires_dstate, = attributes
    # t_diff = t_next - t_prev
    # t_prev = t_prev - 0.001*t_diff
    # t_next = t_next + 0.001*t_diff
    ev_f = []

    def __get_ev_f(__ev, __rds):
        if __rds:
            def __local_ev_f(t):
                return __ev(t, sol(t), sol.grad(t), **consts)
        else:
            def __local_ev_f(t):
                return __ev(t, sol(t), **consts)
        return __local_ev_f

    for ev, rds in zip(events, requires_dstate):
        ev_f.append(__get_ev_f(ev, rds))
    # ev_f = [(lambda event: lambda t: event(t, sol(t), **consts))(ev) for ev in events]

    roots, success = root_finder(
        ev_f,
        [t_prev, t_next],
        tol=None,
        verbose=False
    )

    g = [ev_f[idx](t_root - (t_next - t_prev) * D.epsilon(roots[0].dtype) ** 0.5) for idx, t_root in enumerate(roots)]
    g_cen = [ev_f[idx](t_root) for idx, t_root in enumerate(roots)]
    g_new = [ev_f[idx](t_root + (t_next - t_prev) * D.epsilon(roots[0].dtype) ** 0.5) for idx, t_root in enumerate(roots)]

    g = D.ar_numpy.stack(g)
    g_cen = D.ar_numpy.stack(g_cen)
    g_new = D.ar_numpy.stack(g_new)

    if D.autoray.infer_backend(roots[0]) == 'torch':
        direction = direction.to(g.device)

    up = ((g <= 0) & (g_new >= 0)) | ((g <= 0) & (g_cen >= 0)) | ((g_cen <= 0) & (g_new >= 0))
    down = ((g >= 0) & (g_new <= 0)) | ((g >= 0) & (g_cen <= 0)) | ((g_cen >= 0) & (g_new <= 0))

    for receptive_field in [1.0, 2.0, 3.0]:
        g = [ev_f[idx](t_root - receptive_field * (t_next - t_prev) * D.epsilon(roots[0].dtype) ** 0.75) for idx, t_root in
             enumerate(roots)]
        g_new = [ev_f[idx](t_root + receptive_field * (t_next - t_prev) * D.epsilon(roots[0].dtype) ** 0.75) for idx, t_root in
                 enumerate(roots)]

        g = D.ar_numpy.stack(g)
        g_new = D.ar_numpy.stack(g_new)

        up = up | (((g <= 0) & (g_new >= 0)) | ((g <= 0) & (g_cen >= 0)) | ((g_cen <= 0) & (g_new >= 0)))
        down = down | ((g >= 0) & (g_new <= 0)) | ((g >= 0) & (g_cen <= 0)) | ((g_cen >= 0) & (g_new <= 0))

    up = success & up
    down = success & down
    either = up | down

    # print(roots, success, up, down, either, g, g_cen, g_new)
    mask = (up & (direction > 0) |
                down & (direction < 0) |
                either & (direction == 0))
    
    active_events = D.ar_numpy.reshape(D.ar_numpy.nonzero(mask), (-1,))

    roots = roots[active_events]
    evs = [events[idx] for idx in active_events]
    terminate = False

    if len(active_events) > 0:
        order = D.ar_numpy.argsort(D.ar_numpy.sign(t_next - t_prev) * roots)
        active_events = active_events[order]
        roots = roots[order]
        evs = [evs[idx] for idx in order]

        if D.ar_numpy.any(is_terminal[active_events]):
            t = D.ar_numpy.nonzero(is_terminal[active_events])[0][0]
            active_events = active_events[:t + 1]
            roots = roots[:t + 1]
            evs = evs[:t + 1]
            terminate = True

    return active_events, roots, terminate, evs


class DenseOutput(object):
    """Dense Output class for storing the dense output from a numerical integration.
    
    Attributes
    ----------
    t_eval : list or array-type
        time of evaluation of differential equations
    y_interpolants : list of interpolants
        interpolants of each timestep in a numerical integration
    """

    def __init__(self, t_eval, y_interpolants):
        if t_eval is None and y_interpolants is None:
            self.t_eval = None
            self.__t_eval_arr = None
            self.__t_eval_arr_stale = False
            self.y_interpolants = []
        else:
            if t_eval is None or y_interpolants is None:
                raise ValueError("Both t_eval and y_interpolants must not be NoneTypes")
            elif len(t_eval) != len(y_interpolants) + 1:
                raise ValueError("The number of evaluation times and interpolants must be equal!")
            else:
                self.t_eval = [D.ar_numpy.asarray(t) for t in t_eval]
                self.__t_eval_arr = D.ar_numpy.stack(self.t_eval)
                self.__t_eval_arr_stale = False
                self.y_interpolants = y_interpolants

    @property
    def t_eval_arr(self):
        if self.__t_eval_arr_stale:
            self.__t_eval_arr = D.ar_numpy.stack(self.t_eval)
            self.__t_eval_arr_stale = False
        return self.__t_eval_arr

    def find_interval(self, t):
        if self.t_eval is None:
            raise ValueError("No interpolant has been added and time interval is not defined!")
        return min(deutil.search_bisection(self.t_eval, t), len(self.y_interpolants) - 1)

    def find_interval_vec(self, t):
        if self.t_eval is None:
            raise ValueError("No interpolant has been added and time interval is not defined!")
        out = deutil.search_bisection_vec(self.t_eval_arr, t)
        out[out > len(self.y_interpolants) - 1] = len(self.y_interpolants) - 1
        return out

    def __call__(self, t):
        if len(D.ar_numpy.shape(t)) > 0:
            __flat_t = D.ar_numpy.reshape(D.ar_numpy.asarray(t), (-1,))
            __indices = self.find_interval_vec(__flat_t)
            y_vals = D.ar_numpy.stack([
                self.y_interpolants[idx](_t) for idx, _t in zip(__indices, __flat_t)
            ], axis=0)
            return D.ar_numpy.reshape(y_vals, D.ar_numpy.shape(t) + D.ar_numpy.shape(y_vals)[1:])
        else:
            tidx = self.find_interval(t)
            return self.y_interpolants[tidx](t)

    def grad(self, t):
        if len(D.ar_numpy.shape(t)) > 0:
            __flat_t = D.ar_numpy.reshape(D.ar_numpy.asarray(t), (-1,))
            __indices = self.find_interval_vec(__flat_t)
            y_vals = D.ar_numpy.stack([
                self.y_interpolants[idx].grad(_t) for idx, _t in zip(__indices, __flat_t)
            ], axis=0)
            print(__indices)
            return D.ar_numpy.reshape(y_vals, D.ar_numpy.shape(t) + D.ar_numpy.shape(y_vals)[1:])
        else:
            tidx = self.find_interval(t)
            # print(tidx, self.y_interpolants[tidx])
            return self.y_interpolants[tidx].grad(t)

    def add_interpolant(self, t, y_interp):
        if isinstance(t, list) and isinstance(y_interp, list):
            assert (len(t) == len(y_interp))
            for idx in range(len(t)):
                self.add_interpolant(t[idx], y_interp[idx])
        elif (isinstance(t, list) and not isinstance(y_interp, list)) or (
                not isinstance(t, list) and isinstance(y_interp, list)):
            raise TypeError(
                "Expected both t and y_interp to be lists, but got type(t)={}, type(y_interp)={}".format(type(t), type(
                    y_interp)))
        else:
            try:
                y_interp(t)
            except:
                raise
            if self.t_eval is None or len(self.t_eval) == 0:
                self.t_eval = [D.ar_numpy.asarray(t)]
                self.y_interpolants = [y_interp]
            else:
                try:
                    y_interp(self.t_eval[-1])
                except:
                    raise
                if (t - self.t_eval[-1]) < 0:
                    self.t_eval.insert(0, D.ar_numpy.asarray(t))
                    self.y_interpolants.insert(0, y_interp)
                else:
                    self.t_eval.append(D.ar_numpy.asarray(t))
                    self.y_interpolants.append(y_interp)
            if D.autoray.infer_backend(t) == 'torch':
                self.t_eval = [i.to(D.ar_numpy.asarray(t)) for i in self.t_eval]
            self.__t_eval_arr_stale = True

    def remove_interpolant(self, idx):
        out = self.t_eval.pop(idx), self.y_interpolants.pop(idx)
        self.__t_eval_arr = D.ar_numpy.stack(self.t_eval)
        return out

    def __len__(self):
        return 0 if self.t_eval is None else len(self.t_eval)

    @property
    def t_min(self):
        return D.ar_numpy.min(self.__t_eval_arr)

    @property
    def t_max(self):
        return D.ar_numpy.max(self.__t_eval_arr)


class DiffRHS(object):
    """Differential Equation class. Designed to wrap around around a function for the right-hand side of an ordinary differential equation.
    
    Attributes
    ----------
    rhs : callable
        Right-hand side of an ordinary differential equation
    equ_repr : str
        String representation of the right-hand side.
    """

    def __init__(self, rhs, sample_input=None, equ_repr=None, md_repr=None):
        """Initialises the equation class possibly with a human-readable equation representation.
        
        Parameters
        ----------
        rhs : callable
            A function for obtaining the rhs of an ordinary differential equation with the invocation patter rhs(t, y, **kwargs)
        equ_repr : str, optional
            A human-readable transcription of the differential equation represented by rhs
        """
        self.rhs = rhs
        if equ_repr is not None:
            self.equ_repr = str(equ_repr)
            if md_repr is not None:
                self.md_repr = md_repr
            else:
                self.md_repr = equ_repr
        else:
            if md_repr is not None:
                self.md_repr = md_repr
            else:
                try:
                    self.md_repr = self.rhs._repr_markdown_()
                except:
                    self.md_repr = str(self.rhs)
            self.equ_repr = str(self.rhs)
        self.__jac_wrapped_rhs_order = None
        self.__jac_initialised = False
        self.__jac = None
        self.__jac_time = None
        self.__jac_is_wrapped_rhs = False
        self.nfev = 0
        self.njev = 0

    def __call__(self, t, y, *args, **kwargs):
        """Calls the equation represented by self.rhs with the given arguments.
        Tracks number of function evaluations.
        """
        called_val = self.rhs(t, y, *args, **kwargs)
        self.nfev += 1
        return called_val

    def jac(self, t, y, *args, **kwargs):
        """Returns the jacobian of self.rhs at a given time (t) and state (y).
        Tracks number of jacobian evaluations.
        
        Uses a Richardson Extrapolated 5th order central finite differencing method
        when a jacobian is not defined as self.rhs.jac or by self.hook_jacobian_call
        """
        if not self.__jac_initialised:
            if y is None:
                inferred_backend = "numpy"
            else:
                inferred_backend = D.autoray.infer_backend(y)
            if self.__jac is None:
                if hasattr(self.rhs, 'jac'):
                    self.__jac = self.rhs.jac
                    self.__jac_time = None
                    self.__jac_is_wrapped_rhs = False
                elif inferred_backend == 'numpy':
                    self.__jac_wrapped_rhs_order = 5
                    self.__jac = deutil.JacobianWrapper(lambda y, **kwargs: self(0.0, y, **kwargs),
                                                        base_order=self.__jac_wrapped_rhs_order, flat=False)
                    self.__jac_time = 0.0
                    self.__jac_is_wrapped_rhs = True
                else:
                    import torch
                    self.__jac = torch.func.jacrev(self.rhs, argnums=1)
                    self.__jac_time = None
                    self.__jac_is_wrapped_rhs = False
            self.__jac_initialised = True
        if self.__jac_is_wrapped_rhs:
            if t != self.__jac_time:
                self.__jac_time = t
                self.__jac = deutil.JacobianWrapper(lambda y, **kwargs: self(t, y, **kwargs),
                                                    base_order=self.__jac_wrapped_rhs_order, flat=False)
            called_val = self.__jac(y, *args, **kwargs)
        else:
            called_val = self.__jac(t, y, *args, **kwargs)
        self.njev += 1
        return called_val

    @property
    def jac_is_wrapped_rhs(self):
        return self.__jac_is_wrapped_rhs

    def hook_jacobian_call(self, jac_fn):
        """Attaches a function, jac_fn, that returns the jacobian of self.rhs
        as an array with shape (state.numel(), rhs.numel()).
        """
        self.__jac = jac_fn
        self.__jac_time = None
        self.__jac_is_wrapped_rhs = False

    def unhook_jacobian_call(self):
        """Detaches the current jacobian function. On next call to `jac`
        this will be reinitialised to either a finite difference estimate
        or a pytorch jacrev functional transform        
        """
        self.__jac = None

    def set_jac_base_order(self, order):
        if self.__jac_is_wrapped_rhs:
            self.__jac_wrapped_rhs_order = order
            self.__jac = deutil.JacobianWrapper(lambda y, **kwargs: self.rhs(0.0, y, **kwargs),
                                                base_order=self.__jac_wrapped_rhs_order, flat=True)
            self.__jac_time = 0.0

    def __str__(self):
        return self.equ_repr

    def _repr_markdown_(self):
        return self.md_repr

    def __repr__(self):
        return "<DiffRHS({},{},{})>".format(repr(self.rhs), self.equ_repr, self.md_repr)

    def __copy__(self):
        __new_diff_rhs = DiffRHS(self.rhs, self.equ_repr, self.md_repr)
        if not self.jac_is_wrapped_rhs:
            __new_diff_rhs.hook_jacobian_call(self.__jac)
        return __new_diff_rhs

    def __deepcopy__(self, memo):
        import copy
        __new_diff_rhs = DiffRHS(copy.deepcopy(self.rhs, memo), copy.deepcopy(self.equ_repr, memo),
                                 copy.deepcopy(self.md_repr, memo))
        if not self.jac_is_wrapped_rhs:
            __new_diff_rhs.hook_jacobian_call(copy.deepcopy(self.__jac, memo))
        return __new_diff_rhs

    __base_attributes = ["rhs", "equ_repr", "md_repr", "nfev", "njev", "__jac", "__jac_is_wrapped_rhs",
                         "__jac_wrapped_rhs_order", "__jac_time"]

    def __getattr__(self, name):
        return getattr(self.rhs, name)

    def __setattr__(self, name, val):
        if name.replace("_DiffRHS", "") in self.__base_attributes:
            self.__dict__[name] = val
        elif name == "jac":
            self.hook_jacobian_call(val)
        else:
            self.rhs.__dict__[name] = val


def rhs_prettifier(equ_repr=None, md_repr=None):
    def rhs_wrapper(rhs):
        return DiffRHS(rhs, equ_repr, md_repr)

    return rhs_wrapper


class OdeSystem(object):
    """Ordinary Differential Equation class. Designed to be used with a system of ordinary differential equations."""

    def __init__(self, equ_rhs, y0, t=(0, 1), dense_output=False, dt=1.0, rtol=None, atol=None, constants=dict()):
        """Initialises the system to the parameters passed or to default values.

        Parameters
        ----------
        equ_rhs : callable
            Specifies the right hand side of the system.
             The calling signature of equ_rhs should be:
                 equ_rhs(t, y, **constants)
             NOTE: To make the output prettier, you can decorate the rhs
                   function with a @rhs_prettifier("Equation representation")
                   call where "Equation Representation" is a text representation
                   of your equation.
        y0 : array-like or float
            Specifies the initial state of the system.
        t : tuple of floats, optional
            A tuple of the form (initial time, final time) aka the integration limits.
        dense_output : bool, optional
            Set to True or False to specify whether or not a dense output for the solution
            should be computed.
        dt : float, optional
            Sets the step-size for the integration, choose a value that is slightly less 
            than the highest frequency changes in value of the solutions to the equations.
        rtol, atol : float, optional
            Denotes the target relative and absolute errors respectively. 
            Only for adaptive methods.
            NOTE: rtol and atol are used in the error computation as
                      err_bound = atol + rtol * abs(y)
                  in the same way as it is used in the scipy routines.
        constants : dict, optional
            Dict of keyword arguments passed to equ_rhs.
        """
        
        if len(t) != 2:
            raise ValueError("Two time bounds are required, only {} were given.".format(len(t)))
        if not callable(equ_rhs):
            raise TypeError("equ_rhs is not callable, please pass a callable object for the right hand side.")

        if isinstance(equ_rhs, DiffRHS):
            import copy
            self.equ_rhs = copy.copy(equ_rhs)
        else:
            if hasattr(equ_rhs, "equ_repr"):
                self.equ_rhs = DiffRHS(equ_rhs.rhs, y0, equ_rhs.equ_repr, equ_rhs.md_repr)
            else:
                self.equ_rhs = DiffRHS(equ_rhs, y0)
                
        
        self.__inferred_backend = D.autoray.infer_backend(y0)
        if self.__inferred_backend == 'torch':
            self.device = y0.device
        else:
            self.device = None
            
        self.__array_con_kwargs = dict(dtype=y0.dtype, like=self.__inferred_backend)
        if self.__inferred_backend == 'torch':
            self.__array_con_kwargs['device'] = self.device

        self.__rtol = rtol
        self.__atol = atol
        self.__consts = constants if constants is not None else dict()
        self.__y = D.ar_numpy.copy(y0)[None]
        self.__t = D.ar_numpy.asarray(t[0], **self.__array_con_kwargs)[None]
        self.dim = D.ar_numpy.shape(self.__y[0])
        self.counter = 0
        self.__t0 = D.ar_numpy.asarray(t[0], **self.__array_con_kwargs)
        self.__tf = D.ar_numpy.asarray(t[1], **self.__array_con_kwargs)
        self.__method = integrators.RK45CKSolver
        self.integrator = None
        self.__dt = D.ar_numpy.asarray(dt, **self.__array_con_kwargs)
        self.__dt0 = self.dt

        self.staggered_mask = None
        self.__dense_output = dense_output
        self.__int_status = 0
        self.__sol = DenseOutput(None, None)

        self.__move_to_device()
        self.__allocate_soln_space(self.__alloc_space_steps(self.tf))
        self.__fix_dt_dir(self.tf, self.t0)
        self.__events = []
        self.initialise_integrator(preserve_states=False)

    @property
    def sol(self):
        if self.__dense_output:
            return self.__sol
        else:
            return None

    @property
    def success(self):
        return self.__int_status == 1 or self.__int_status == 2

    @property
    def events(self):
        """A tuple of (time, state, event) tuples containing the times and states
        for each event that triggered
        
        Examples
        --------

        >>> ode_system = desolver.OdeSystem(...)
        >>> ode_system.integrate(events=[...])
        >>> ode_system.events
        (StateTuple(t=..., y=..., event=event_1), StateTuple(t=..., y=..., event=event_2), etc.)        
        """
        return self.__events

    @property
    def events_dict(self):
        """A tuple of (time, state, event) tuples containing the times and states
        for each event that triggered
        
        Examples
        --------

        >>> ode_system = desolver.OdeSystem(...)
        >>> ode_system.integrate(events=[...])
        >>> ode_system.events
        (StateTuple(t=..., y=..., event=event_1), StateTuple(t=..., y=..., event=event_2), etc.)        
        """
        event_fn = set([i.event for i in self.__events])
        return {
            ev: StateTuple(
                t=D.ar_numpy.stack([i.t for i in self.__events if i.event == ev]),
                y=D.ar_numpy.stack([i.y for i in self.__events if i.event == ev]),
                event=ev
            ) for ev in event_fn
        }

    @property
    def y(self):
        """The states at which the system has been evaluated.
        """
        return self.__y[:self.counter + 1]

    @property
    def t(self):
        """The times at which the system has been evaluated.
        """
        return self.__t[:self.counter + 1]

    @property
    def nfev(self):
        """The number of function evaluations used during the numerical integration
        """
        return self.equ_rhs.nfev

    @property
    def njev(self):
        """The number of jacobian evaluations used during the numerical integration
        """
        return self.equ_rhs.njev

    @property
    def constants(self):
        """A dictionary of constants for the differential system.
        """
        return self.__consts

    @constants.setter
    def constants(self, new_constants):
        """Sets the constants in the differential system
        """
        self.__consts = new_constants

    @constants.deleter
    def constants(self):
        self.__consts = dict()

    @property
    def rtol(self):
        """The relative tolerance of the adaptive integration schemes
        """
        return self.__rtol

    @rtol.setter
    def rtol(self, new_rtol):
        """Sets the target relative error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive.
        """
        self.__rtol = D.ar_numpy.asarray(new_rtol, **self.__array_con_kwargs)
        self.initialise_integrator()

    @property
    def atol(self):
        """The absolute tolerance of the adaptive integration schemes
        """
        return self.__atol

    @atol.setter
    def atol(self, new_atol):
        """Sets the target absolute error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive.
        """
        self.__atol = D.ar_numpy.asarray(new_atol, **self.__array_con_kwargs)
        self.initialise_integrator()

    @property
    def dt(self):
        """The timestep of the numerical integration
        """
        return self.__dt

    @dt.setter
    def dt(self, new_dt):
        self.__dt = D.ar_numpy.asarray(new_dt, **self.__array_con_kwargs)
        self.__move_to_device()
        self.__fix_dt_dir(self.tf, self.t0)
        return self.__dt

    @property
    def t0(self):
        """The initial integration time
        """
        return self.__t0

    @t0.setter
    def t0(self, new_t0):
        """Changes the initial time for the integration of the ODE system.

        Parameters
        ----------
        new_t0 : float
            Initial integration time.
            
        Raises
        ------
        ValueError
            If the initial integration time is greater than the final time and the integration 
            has been run (successfully or unsuccessfully).
        """
        new_t0 = D.ar_numpy.asarray(new_t0, **self.__array_con_kwargs)
        if D.ar_numpy.abs(self.tf - new_t0) <= D.epsilon(self.__y[self.counter].dtype):
            raise ValueError("The start time of the integration cannot be greater than or equal to {}!".format(self.tf))
        self.__t0 = new_t0
        self.__move_to_device()
        self.__fix_dt_dir(self.tf, self.t0)

    @property
    def tf(self):
        """The final integration time
        """
        return self.__tf

    @tf.setter
    def tf(self, new_tf):
        """Changes the initial time for the integration of the ODE system.

        Parameters
        ----------
        new_tf : float
            Initial integration time.
            
        Raises
        ------
        ValueError
            If the initial integration time is greater than the final time and the integration 
            has been run (successfully or unsuccessfully).
        """
        new_tf = D.ar_numpy.asarray(new_tf, **self.__array_con_kwargs)
        if D.ar_numpy.abs(self.t0 - new_tf) <= D.epsilon(self.__y[self.counter].dtype):
            raise ValueError("The end time of the integration cannot be equal to the start time: {}!".format(self.t0))
        self.__tf = new_tf
        self.__move_to_device()
        self.__fix_dt_dir(self.tf, self.t0)

    def __fix_dt_dir(self, t1, t0):
        if D.ar_numpy.sign(self.__dt) != D.ar_numpy.sign(t1 - t0):
            self.__dt = -self.__dt
        else:
            self.__dt = self.__dt

    def __move_to_device(self):
        if self.device is not None and self.__inferred_backend == 'torch':
            self.__y[0] = self.__y[0].to(self.device)
            self.__t[0] = self.__t[0].to(self.device)
            self.__t0 = self.__t0.to(self.device)
            self.__tf = self.__tf.to(self.device)
            self.__dt = self.__dt.to(self.device)
            self.__dt0 = self.__dt0.to(self.device)
            try:
                self.__atol = self.__atol.to(self.device)
            except AttributeError:
                pass
            except:
                raise
            try:
                self.__rtol = self.__rtol.to(self.device)
            except AttributeError:
                pass
            except:
                raise
            try:
                self.equ_rhs.rhs = self.equ_rhs.rhs.to(self.device)
            except AttributeError:
                pass
            except:
                raise
        return

    def __alloc_space_steps(self, tf):
        """Returns the number of steps to allocate for a given final integration time
        
        Returns
        -------
        int
            integer number of steps to allocate in the solution arrays of y and t. Defaults to 10 if the final time is set to infinity.
        """
        if D.ar_numpy.to_numpy(tf) == np.inf:
            return 10
        else:
            return max(1, min(5000, int((tf - self.__t[self.counter]) / self.dt)))

    def __allocate_soln_space(self, num_units):
        try:
            if num_units != 0:
                if self.__inferred_backend in ['numpy', 'torch']:
                    __new_allocs = D.ar_numpy.zeros((num_units,) + D.ar_numpy.shape(self.__y[0]), **self.__array_con_kwargs)
                    self.__y = D.ar_numpy.concatenate(
                        [self.__y, __new_allocs], axis=0)
                    
                    __new_allocs = D.ar_numpy.zeros((num_units,) + D.ar_numpy.shape(self.__t[0]), **self.__array_con_kwargs)
                    self.__t = D.ar_numpy.concatenate(
                        [self.__t, __new_allocs], axis=0)
                else:
                    self.__y = self.__y + [None for _ in range(num_units)]
                    self.__t = self.__t + [None for _ in range(num_units)]
        except MemoryError as E:
            deutil.warning("Final tf ({}) too large, space allocation failed with MemoryError:\n{}".format(self.tf, E))
            self.__allocate_soln_space(100)

    def __trim_soln_space(self):
        self.__y = self.__y[:self.counter + 1]
        self.__t = self.__t[:self.counter + 1]

    def set_kick_vars(self, staggered_mask):
        """Sets the variable mask for the symplectic integrators. 

        The conventional structure of a symplectic integrator is Kick-Drift-Kick
        or Drift-Kick-Drift where Drift is when the Positions are updated and
        Kick is when the Velocities are updated.

        The default assumption is that the latter half of the variables are
        to be updated in the Kick step and the former half in the Drift step.

        Does nothing if integrator is not symplectic.
        
        Parameters
        ----------
        staggered_mask : array of bools
            A boolean array with the same shape as y. Specifies the elements 
            of y that are to be updated as part of the Kick step.
        """
        self.staggered_mask = staggered_mask
        self.initialise_integrator()

    def get_current_time(self):
        """Returns the current time of the ODE system"""
        return self.t[self.counter]

    def initialise_integrator(self, preserve_states=False):
        with D.autoray.backend_like(self.y[0]):
            integrator_kwargs = dict(dtype=self.y[-1].dtype, device=self.device)

            integrator_kwargs['atol'] = self.atol
            integrator_kwargs['rtol'] = self.rtol

            if self.method.symplectic and not self.method.is_implicit:
                integrator_kwargs['staggered_mask'] = self.staggered_mask

            if self.integrator:
                old_states_exist = True
                old_dState = self.integrator.dState
                old_dTime = self.integrator.dTime
            else:
                old_states_exist = False
                old_dState = None
                old_dTime = None
            self.integrator = self.method(self.dim, **integrator_kwargs)
            if old_states_exist and preserve_states:
                self.integrator.dState = old_dState
                self.integrator.dTime = old_dTime
                if self.__inferred_backend == 'torch':
                    self.integrator.dState = self.integrator.dState.to(self.integrator.device)
                    self.integrator.dTime = self.integrator.dTime.to(self.integrator.device)

    def __get_integrator_mask(self, staggered_mask):
        if staggered_mask is None and hasattr(self.integrator, "staggered_mask"):
            return self.integrator.staggered_mask
        return staggered_mask

    @property
    def method(self):
        """The numerical integration scheme
        """
        return self.__method

    @method.setter
    def method(self, new_method):
        self.set_method(new_method, None)

    def set_method(self, new_method, staggered_mask=None, preserve_states=True):
        """Sets the method of integration.

        Parameters
        ---------
        new_method : str or stateful functor
            A string that is the name of an integrator that is available in DESolver, OR a functor that 
            takes the current time, state, time step and equation, and advances the state by 1 timestep 
            adjusting the timestep as necessary. 
            
        staggered_mask : boolean masking array
            A boolean array with the same shape as the system state that indicates which variables are
            updated during the 'kick' stage of a symplectic integrator. Has no effect if the integrator
            is adaptive.
            
        Raises
        ------
        ValueError
            If the string is not a valid integration scheme.
        """
        self.staggered_mask = self.__get_integrator_mask(staggered_mask)
        if self.__int_status == 1:
            deutil.warning("An integration was already run, this will not reset the system", category=RuntimeWarning)
        if new_method in integrators.available_methods():
            self.__method = integrators.available_methods(False)[new_method]
        elif issubclass(new_method, integrators.IntegratorTemplate):
            if not issubclass(new_method, integrators.RichardsonIntegratorTemplate) and new_method not in map(
                    lambda x: x[1], integrators.available_methods(False).items()):
                deutil.warning(
                    "This is not a method implemented as part of the DESolver package. Cannot guarantee results.")
            self.__method = new_method
        else:
            raise ValueError("The method you selected does not exist in the list of available methods, \
                              call desolver.available_methods() to see what these are")
        self.initialise_integrator(preserve_states=preserve_states)

    def get_step_interpolant(self):
        return self.integrator.dense_output()

    @property
    def integration_status(self):
        """Returns the integration status as a human-readable string.

        Returns
        -------
        str
            String containing the integration status message.
        """
        if self.__int_status == 0:
            return "Integration has not been run."
        if isinstance(self.__int_status, KeyboardInterrupt):
            return "A KeyboardInterrupt exception was raised during integration."
        if isinstance(self.__int_status, etypes.FailedIntegration):
            if self.__int_status.__cause__ is not None:
                return "The integration failed with the following exception: {}\n\tCaused by {}".format(
                    self.__int_status, self.__int_status.__cause__)
            else:
                return "The integration failed with the following exception: {}".format(self.__int_status)
        if self.__int_status == 1:
            return "Integration completed successfully."
        if self.__int_status == 2:
            return "Integration terminated upon finding a triggered event."
        else:
            return "It should not be possible to initialise the system with this status"

    def reset(self):
        """Resets the system to the initial time."""
        self.counter = 0
        self.__trim_soln_space()
        self.__sol = DenseOutput(None, None)
        self.dt = self.__dt0
        self.equ_rhs.nfev = 0
        self.__move_to_device()
        self.__int_status = 0
        if self.__events:
            self.__events = []
        self.initialise_integrator(preserve_states=False)

    def integrate(self, t=None, callback=None, eta=False, events=None):
        """Integrates the system to a specified time.

        Parameters
        ----------
        t : float
            If t is specified, then the system will be integrated to time t. Otherwise the system will 
            integrate to the specified final time. 
            NOTE: t can be negative in order to integrate backwards in time, but use this with caution as this functionality is slightly unstable.
                  
        callback : callable or list of callables
            A callable object or list of callable objects that are invoked as callback(self) at each time step.
            e.g. for logging integration to disk, saving data, manipulating the state of the system, etc.
                  
        eta : bool
            Specifies whether or not the integration process should return an eta, current progress 
            and simple information regarding step-size and current time. Will be deprecated
            in the future in favour of verbosity argument that prints once every n-steps.
            NOTE: This may slow the integration process down as the process of outputting these values create overhead.
                  
        events : callable or list of callables
            Events to track, defaults to None. Each function must have the signature ``event(t, y, **kwargs)``
            and the solver will find the time t such that ``event(t, y, **kwargs) == 0``. The ``**kwargs`` argument
            allows the solver to pass the system constants to the function.
            Additionally, each event function can possess the following two attributes:
            
                direction: bool, optional
                    Indicates the direction of the event crossing that will register an event.
                is_terminal: bool, optional
                    Indicates whether the detection of the event terminates the numerical integration.
                  
        Raises
        ------
        RecursionError : 
            Raised if an adaptive integrator recurses beyond the recursion limit when attempting to compute a forward step.
            This usually means that the numerical integration did not converge and that the 
            behaviour of the system is highly unreliable. This could be due to numerical issues.
            
        """
        if t:
            tf = t
        else:
            tf = self.tf

        if D.ar_numpy.abs(tf - self.__t[self.counter]) < D.epsilon(self.__y[self.counter].dtype):
            return
        steps = 0

        events, is_terminal, direction, last_occurrence, requires_dstate = prepare_events(events, self.__y[0])

        implicit_integration = False
        if D.ar_numpy.to_numpy(tf) == np.inf:
            implicit_integration = True
            if not any(is_terminal):
                deutil.warning(
                    "Specifying an indefinite integration time with no terminal events "
                    "can lead to memory issues if no event terminates the integration.",
                    category=RuntimeWarning)

        self.__fix_dt_dir(tf, self.__t[self.counter])

        if D.ar_numpy.abs(self.dt) > D.ar_numpy.abs(tf - self.__t[self.counter]):
            self.dt = D.ar_numpy.abs(tf - self.__t[self.counter]) * 0.5

        total_steps = self.__alloc_space_steps(tf)

        if eta:
            if tf == np.inf:
                tqdm_progress_bar = tqdm(total=None)
            else:
                tqdm_progress_bar = tqdm(total=int((tf - self.__t[self.counter]) / self.dt) + 1)
        else:
            tqdm_progress_bar = None

        try:
            callback = list(callback)
        except:
            if callback is None:
                callback = []
            else:
                callback = [callback]

        end_int = False
        cState = D.ar_numpy.zeros_like(self.__y[self.counter])
        cTime = D.ar_numpy.zeros_like(self.__t[self.counter])
        self.__allocate_soln_space(total_steps)
        try:
            while (implicit_integration or self.dt != 0 and D.ar_numpy.abs(tf - self.__t[self.counter]) >= D.tol_epsilon(self.__y[self.counter].dtype)) and not end_int:
                if not implicit_integration and D.ar_numpy.abs(self.dt + self.__t[self.counter]) > D.ar_numpy.abs(tf):
                    self.dt = (tf - self.__t[self.counter])
                self.dt, (dTime, dState) = self.integrator(self.equ_rhs, self.__t[self.counter], self.__y[self.counter],
                                                           self.constants, timestep=self.dt)

                if self.counter + 1 >= len(self.__y):
                    total_steps = self.__alloc_space_steps(tf - dTime) + 1
                    self.__allocate_soln_space(total_steps)

                #
                # Compensated Summation based on 
                # https://reference.wolfram.com/language/tutorial/NDSolveSPRK.html
                #

                dState = dState + cState
                dTime = dTime + cTime

                self.__y[self.counter + 1] = self.__y[self.counter] + dState
                self.__t[self.counter + 1] = self.__t[self.counter] + dTime

                cState = (self.__y[self.counter + 1] - self.__y[self.counter]) - dState
                cTime = (self.__t[self.counter + 1] - self.__t[self.counter]) - dTime

                self.counter += 1

                if events is not None or self.__dense_output:
                    __pre_length = len(self.__sol)
                    __t_interp, __y_interp = self.get_step_interpolant()
                    self.__sol.add_interpolant(__t_interp, __y_interp)

                    if events is not None:
                        next_time = self.__t[self.counter]
                        next_state = self.__y[self.counter]
                        prev_time = self.__t[self.counter - 1]
                        prev_state = self.__y[self.counter - 1]
                        self.counter -= 1

                        sol_tuple = (self.__sol, prev_time, next_time)
                        active_events, roots, end_int, evs = handle_events(sol_tuple, events, self.constants, direction, is_terminal, (requires_dstate,))

                        if self.counter + len(roots) + 1 >= len(self.__y):
                            total_steps = self.__alloc_space_steps(tf - dTime) + 1 + len(roots)
                            self.__allocate_soln_space(total_steps)

                        for ev_idx, (root, ev) in enumerate(zip(roots, evs)):
                            if dTime >= 0:
                                true_positive = (self.__t[self.counter] <= root) & (root <= prev_time + dTime)
                            else:
                                true_positive = (prev_time + dTime <= root) & (root <= self.__t[self.counter])

                            if true_positive:
                                ev_state = StateTuple(t=root, y=self.__sol(root), event=ev)
                                if not self.__events or last_occurrence[ev_idx] == -1:
                                    last_occurrence[ev_idx] = len(self.__events)
                                    self.__events.append(ev_state)
                                elif D.ar_numpy.abs(ev_state.t - self.__events[last_occurrence[ev_idx]].t) > D.epsilon(self.__y[0].dtype) ** 0.7:
                                    last_occurrence[ev_idx] = len(self.__events)
                                    self.__events.append(ev_state)

                        if end_int:
                            self.integrate(roots[-1])
                            self.__int_status = 2
                        else:
                            if self.counter + len(roots) + 1 >= len(self.__y):
                                total_steps = self.__alloc_space_steps(tf - dTime) + 1 + len(roots)
                                self.__allocate_soln_space(total_steps)
                            self.__t[self.counter + 1] = prev_time + dTime
                            self.__y[self.counter + 1] = prev_state + dState
                            self.counter += 1

                    if not self.__dense_output:
                        for _ in range(__pre_length - 1):
                            self.__sol.remove_interpolant(0)

                steps += 1

                for i in callback:
                    i(self)

                if tqdm_progress_bar is not None:
                    tqdm_progress_bar.total = tqdm_progress_bar.n
                    if D.ar_numpy.to_numpy(tf) == np.inf:
                        tqdm_progress_bar.total = None
                    else:
                        tqdm_progress_bar.total = tqdm_progress_bar.n + int((tf - self.__t[self.counter]) / self.dt) + 1
                    tqdm_progress_bar.desc = "{:>10.2f} | {:.2f} | {:<10.2e}".format(self.__t[self.counter], tf,
                                                                                     self.dt).ljust(8)
                    tqdm_progress_bar.update()

        except KeyboardInterrupt as e:
            self.__int_status = e
            raise e
        except Exception as e:
            new_e = etypes.FailedIntegration("Failed to integrate system")
            new_e.__cause__ = e
            self.__int_status = new_e
            raise new_e
        else:
            if self.__int_status != 2 and not isinstance(self.__int_status,
                                                         (etypes.FailedIntegration, KeyboardInterrupt)):
                self.__int_status = 1
        finally:
            if eta:
                tqdm_progress_bar.close()
            self.__trim_soln_space()

    def __repr__(self):
        return "\n".join([
            """{:>10}: {:<128}""".format("message", self.integration_status),
            """{:>10}: {:<128}""".format("nfev", str(self.nfev)),
            """{:>10}: {:<128}""".format("sol", str(self.sol)),
            """{:>10}: {:<128}""".format("t0", str(self.t0)),
            """{:>10}: {:<128}""".format("tf", str(self.tf)),
            """{:>10}: {:<128}""".format("y0", str(self.y[0])),
            """{:>10}: {}     """.format("Equations", repr(self.equ_rhs)),
            """{:>10}: {:<128}""".format("t", str(self.t)),
            """{:>10}: {:<128}""".format("y", str(self.y)),
        ])

    def _repr_markdown_(self):
        return """```
{:>10}: {:<128}  
{:>10}: {:<128}  
{:>10}: {:<128}  
{:>10}: {:<128}  
{:>10}: {:<128}  
{:>10}: {:<128}  
{:>10}: 
```  
{}
 
```  
{:>10}: {:<128}  
{:>10}: {:<128}  
```
""".format(
            "message", self.integration_status,
            "nfev", str(self.nfev),
            "sol", str(self.sol),
            "t0", str(self.t0),
            "tf", str(self.tf),
            "y0", str(self.y[0]),
            "Equations", self.equ_rhs._repr_markdown_(),
            "t", str(self.t),
            "y", str(self.y))

    def __str__(self):
        """Prints the equations, initial conditions, final states, time limits and defined constants in the system."""
        print_str = "y({t0}) = {init_val}\ndy = {equation}\ny({t}) = {cur_val}\n"
        print_str = print_str.format(init_val=self.y[0],
                                     t0=self.t0,
                                     equation=str(self.equ_rhs),
                                     t=self.t[-1],
                                     cur_val=self.y[-1])
        if self.constants:
            print_str += "\nThe constants that have been defined for this system are: "
            print_str += "\n" + str(self.constants)

        print_str += "\nThe time limits for this system are:\n"
        print_str += "t0 = {}, tf = {}, t_current = {}, step_size = {}".format(self.t0, self.tf, self.t[-1], self.dt)

        return print_str

    def __getitem__(self, index):
        if isinstance(index, int):
            if index > self.counter:
                raise IndexError(
                    "index {} out of bounds for integrations with {} steps".format(index, self.counter + 1))
            else:
                return StateTuple(t=self.t[index], y=self.y[index], event=None)
        elif isinstance(index, slice):
            if index.start is not None:
                start_idx = deutil.search_bisection(self.t[:self.counter + 1], index.start)
            else:
                start_idx = 0
            if index.stop is not None:
                end_idx = deutil.search_bisection(self.t[:self.counter + 1], index.stop) + 1
            else:
                end_idx = self.counter + 1
            if index.step is not None:
                step = index.step
            else:
                step = 1
            return StateTuple(t=self.t[start_idx:end_idx:step], y=self.y[start_idx:end_idx:step], event=None)
        else:
            if self.__dense_output and self.sol is not None:
                return StateTuple(t=index, y=self.sol(index), event=None)
            else:
                nearest_idx = deutil.search_bisection(self.__t, index)
                if nearest_idx < self.counter:
                    if D.ar_numpy.abs(D.ar_numpy.to_numpy(self.t[nearest_idx] - index)) < D.ar_numpy.abs(
                            D.ar_numpy.to_numpy(self.t[nearest_idx + 1] - index)):
                        return StateTuple(t=self.t[nearest_idx], y=self.y[nearest_idx], event=None)
                    else:
                        return StateTuple(t=self.t[nearest_idx + 1], y=self.y[nearest_idx + 1], event=None)
                else:
                    return StateTuple(t=self.t[nearest_idx], y=self.y[nearest_idx], event=None)

    def __len__(self):
        return self.counter + 1
