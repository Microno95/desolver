"""
The MIT License (MIT)

Copyright (c) 2019 Microno95, Ekin Ozturk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from scipy.interpolate import CubicSpline

from tqdm.auto import tqdm

from . import backend as D
from . import integration_schemes as ischemes
from . import exception_types as etypes
from . import utilities as deutil

__all__ = [
    'DiffRHS',
    'rhs_prettifier',
    'OdeSystem'
]

class DiffRHS(object):
    """Differential Equation class. Designed to wrap around around a function for the right-hand side of an ordinary differential equation.
    
    Attributes
    ----------
    rhs : callable
        Right-hand side of an ordinary differential equation
    equ_repr : str
        String representation of the right-hand side.
    """
    def __init__(self, rhs, equ_repr=None):
        """Initialises the equation class possibly with a human-readable equation representation.
        
        Parameters
        ----------
        rhs : callable
            A function for obtaining the rhs of an ordinary differential equation with the invocation patter rhs(t, y, **kwargs)
        equ_repr : str, optional
            A human-readable transcription of the differential equation represented by rhs
        """
        self.rhs     = rhs
        if equ_repr is not None:
            self.equ_repr = str(equ_repr)
        else:
            self.equ_repr = str(self.rhs)
        
    def __call__(self, t, y, **kwargs):
        return self.rhs(t, y, **kwargs)

    def __str__(self):
        return self.equ_repr

    def __repr__(self):
        return "<DiffRHS({},{})>".format(repr(self.rhs), self.equ_repr)

def rhs_prettifier(equRepr):
    def rhs_wrapper(rhs):
        return DiffRHS(rhs, equRepr)
    return rhs_wrapper

class OdeSystem(object):
    """Ordinary Differential Equation class. Designed to be used with a system of ordinary differential equations."""
    def __init__(self, equ_rhs, y0, t=(0, 1), dense_output=False, dt=1.0, rtol=1e-6, atol=1e-6, constants=dict()):
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
            Set to True or False to specify whether or not a cubic spline fit should be 
            computed for the integration.
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
            raise etypes.LengthError("Two time bounds are required, only {} were given.".format(len(t)))
        if not hasattr(equ_rhs, "__call__"):
            raise TypeError("equ_rhs is not callable, please pass a callable object for the right hand side.")
        self.nfev = 0
        
        def equ_rhs_wrapped(*args, **kwargs):
            self.nfev += 1
            return equ_rhs(*args, **kwargs)
        
        try:
            self.equ_rhs     = DiffRHS(equ_rhs_wrapped, equ_rhs.equRepr)
        except AttributeError:
            self.equ_rhs     = DiffRHS(equ_rhs_wrapped)
        except:
            raise
        self.rtol        = rtol
        self.atol        = atol
        self.consts      = constants if constants is not None else dict()
        self._y          = [D.copy(y0)]
        self._t          = [D.to_float(t[0])]
        self.dim         = D.shape(self._y[0])
        self.counter     = 0
        self.t0          = D.to_float(t[0])
        self.t1          = D.to_float(t[1])
        self.method      = ischemes.available_methods["RK45CK"]
        self.integrator  = None
        self.dt          = D.to_float(dt)
        self.__fix_dt_dir(self.t1, self.t0)
        self.dt0         = self.dt
        
        if D.backend() == 'torch':
            self.device = y0.device
        else:
            self.device = None
            
        self.staggered_mask = None
        self.dense_output   = dense_output
        self.int_status     = 0
        self.success        = False
        self.sol            = None
            
        self.__move_to_device()
        self.__allocate_soln_space(10)
        self.initialise_integrator()

    @property
    def y(self):
        return self._y[:self.counter + 1]
    
    @property
    def t(self):
        return self._t[:self.counter + 1]
        
    def __fix_dt_dir(self, t1, t0):
        if D.sign(self.dt) != D.sign(t1 - t0):
            self.dt      = -self.dt
        else:
            self.dt      =  self.dt
        
    def __move_to_device(self):
        if self.device is not None and D.backend() == 'torch':
            self._y[0]  = self.y[0].to(self.device)
            self._t[0]  = self.t[0].to(self.device)
            self.t0     = self.t0.to(self.device)
            self.t1     = self.t1.to(self.device)
            self.dt     = self.dt.to(self.device)
            self.dt0    = self.dt0.to(self.device)
        return
        
    def __allocate_soln_space(self, num_units):
        if D.backend() == 'numpy':
            if num_units == 0:
                self._y = D.stack(self._y)
                self._t = D.stack(self._t)
            else:
                self._y  = D.concatenate([self._y, D.zeros((num_units, *D.shape(self._y[0])), dtype=self._y[0].dtype)], axis=0)
                self._t  = D.concatenate([self._t, D.zeros((num_units, *D.shape(self._t[0])), dtype=self._y[0].dtype)], axis=0)
        else:
            if num_units != 0:
                self._y  = self._y + [None for _ in range(num_units)]
                self._t  = self._t + [None for _ in range(num_units)]
    
    def __trim_soln_space(self):
        self._y = self._y[:self.counter+1]
        self._t = self._t[:self.counter+1]
        
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
        
    def set_end_time(self, tf):
        """Changes the final time for the integration of the ODE system.

        Parameters
        ----------
        tf : float
            Final integration time.
            
        Raises
        ------
        ValueError
            If the final integration time is less than the initial time and the integration 
            has been run (successfully or unsuccessfully).
        """
        if tf <= self.t0 and self.int_status != 0:
            raise ValueError("The end time of the integration cannot be less than "
                             "or equal to {}!".format(self.t0))
        self.t1 = D.to_float(tf)
        self.__move_to_device()
        self.__fix_dt_dir(self.t1, self.t0)

    def get_end_time(self):
        """Returns the final time of the ODE system."""
        return self.t1

    def set_start_time(self, ti):
        """Changes the initial time for the integration of the ODE system.

        Parameters
        ----------
        ti : float
            Initial integration time.
            
        Raises
        ------
        ValueError
            If the initial integration time is greater than the final time and the integration 
            has been run (successfully or unsuccessfully).
        """
        if self.t1 <= ti and self.int_status != 0:
            raise ValueError("The start time of the integration cannot be greater "
                             "than or equal to {}!".format(self.t1))
        self.t0 = D.to_float(ti)
        self.__move_to_device()
        self.__fix_dt_dir(self.t1, self.t0)

    def get_start_time(self):
        """Returns the initial time of the ODE system."""
        return self.t0

    def get_current_time(self):
        """Returns the current time of the ODE system"""
        return self.t[self.counter]

    def set_step_size(self, dt):
        """Sets the step size that will be used for the integration.

        Parameters
        ----------
        dt : float
            Step size value. For systems that grow exponentially choose a smaller value, for oscillatory systems choose
            a value slightly less than the highest frequency of oscillation."""
        self.dt  = D.to_float(dt)
        self.dt0 = self.dt
            
        self.__move_to_device()
        self.__fix_dt_dir(self.t1, self.t0)

    def get_step_size(self):
        """Returns the step size that will be attempted for the next integration step"""
        return self.dt

    def set_rtol(self, new_rtol):
        """Sets the target relative error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive..
        """
        self.rtol = new_rtol

    def get_rtol(self):
        """Returns the target relative error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive..
        """
        return self.rtol

    def set_atol(self, new_atol):
        """Sets the target absolute error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive..
        """
        self.atol = new_atol

    def get_atol(self):
        """Returns the target absolute error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive..
        """
        return self.atol

    def initialise_integrator(self):
        integrator_kwargs = dict(dtype=self.y[0].dtype, device=self.device)
        
        if self.method.__adaptive__:
            integrator_kwargs['atol'] = self.atol
            integrator_kwargs['rtol'] = self.rtol
            
        if self.method.__symplectic__:
            integrator_kwargs['staggered_mask'] = self.staggered_mask
            
        self.integrator = self.method(self.dim, **integrator_kwargs)
        
    def __get_integrator_mask(self, staggered_mask):
        if staggered_mask is None and hasattr(self.integrator, "staggered_mask"):
            return self.integrator.staggered_mask
        return staggered_mask
    
    def set_method(self, method, staggered_mask=None):
        """Sets the method of integration.

        Parameters
        ---------
        method : str or stateful functor
            A string that is the name of an integrator that is available in DESolver, OR a functor that 
            takes the current state, time, time step and equation, and advances the state by 1 timestep 
            adjusting the timestep as necessary. 
            
        Raises
        ------
        ValueError
            If the string is not a valid integration scheme.
        """
        self.staggered_mask = self.__get_integrator_mask(staggered_mask)
        if self.int_status == 1:
            deutil.warning("An integration was already run, the system will be reset")
            self.reset()
        if method in ischemes.available_methods.keys():
            self.method = ischemes.available_methods[method]
        elif issubclass(method, ischemes.IntegratorTemplate):
            if method not in map(lambda x:x[1], ischemes.available_methods.items()):
                deutil.warning("This is not a method implemented as part of the DESolver package. Cannot guarantee results.")
            self.method = method
        else:
            raise ValueError("The method you selected does not exist in the list of available methods, \
                              call desolver.available_methods() to see what these are")
        self.initialise_integrator()

    def add_constants(self, **additional_constants):
        """Takes an arbitrary list of keyword arguments to add to the list of available constants.

        Parameters
        ----------
        additional_constants : keyword arguments, variable-length
            Keyword arguments of constants should be added/updated in the constants associated with the system.
        """
        self.consts.update(additional_constants)

    def remove_constants(self, *constants_removal):
        """Takes an arbitrary list of keyword arguments to add to the list of available constants.

        Parameters
        ----------
        constants_removal : string, list, tuple or dict arguments, variable-length
            Keys of constants that should be removed from the constants list.
        """
        for i in constants_removal:
            if isinstance(i, (list, tuple, dict)):
                self.constants_removal(*list(i))
            elif i in self.consts.keys():
                del self.consts[i]

    def integration_status(self):
        """Returns the integration status as a human-readable string.

        Returns
        -------
        str
            String containing the integration status message.
        """
        if self.int_status == 0:
            return "Integration has not been run."
        elif self.int_status == 1:
            return "Integration completed successfully."
        elif self.int_status == -1:
            return "Recursion limit was reached during integration, "+\
                   "this can be caused by the adaptive integrator being unable "+\
                   "to find a suitable step size to achieve the rtol/atol "+\
                   "requirements."
        elif self.int_status == -2:
            return "A KeyboardInterrupt exception was raised during integration."
        elif self.int_status == -3:
            return "A generic exception was raised during integration."

    def set_dense_output(self, dense_output):
        """Enables/Disables dense output of results.
        
        Parameters
        ----------
        dense_output : bool
            Sets if dense_output should be computed. If a dense output exists, it is discarded.
        """
        if self.sol is not None:
            self.sol = None
        self.dense_output = dense_output

    def compute_dense_output(self):
        """Computes an interpolating CubicSpline over the solution of the integration.
        
        Currently uses numpy arrays and does not permit torch.tensor interpolants.
        
        Returns
        -------
        CubicSpline
            A cubic spline interpolant for the results of the numerical integration.
            
        Raises
        ------
        ValueError
            Raised if a numerical integration was not run.
        FailedIntegrationError
            Raised if the numerical integration was abruptly terminated/did not converge.
        """
        if self.int_status == 0:
            raise ValueError("Cannot compute dense output for non-existent integration.")
        elif self.int_status != 1:
            if self.int_status == -2:
                deutil.warning(
                    "A KeyboardInterrupt was raised during integration,",
                    "the interpolating spline will only be valid between",
                    "{} and {}.".format(self.t0, self.t[-1])
                )
            else:
                raise etypes.FailedIntegrationError("Integration failed with message:"+self.integration_status())
        self.__trim_soln_space()
        self.sol = CubicSpline(D.to_numpy(self.t), D.to_numpy(self.y), extrapolate=True)
        return self.sol

    def reset(self):
        """Resets the system to the initial time."""
        self.counter = 0
        self.__trim_soln_space()
        self.sol     = None
        self.dt      = self.dt0
        self.nfev    = 0
        self.__move_to_device()

    def integrate(self, t=None, callback=None, eta=False):
        """Integrates the system to a specified time.

        Parameters
        ----------
        t : float
            If t is specified, then the system will be integrated to time t. Otherwise the system will 
            integrate to the specified final time. 
            NOTE: t can be negative in order to integrate backwards in time, but use this with caution as this
                  functionality is slightly unstable.
                  
        callback : callable
            A callable object that is invoked as callback(self) at each time step.
            e.g. for logging integration to disk, saving data, manipulating the state of the system, etc.
                  
        eta : bool
            Specifies whether or not the integration process should return an eta, current progress 
            and simple information regarding step-size and current time. Will be deprecated
            in the future in favour of verbosity argument that prints once every n-steps.
            NOTE: This may slow the integration process down as the process of outputting
                  these values create overhead.
                  
        Raises
        ------
        RecursionError
            Raised if an adaptive integrator recurses when attempting to compute a forward step.
            This usually means that the numerical integration did not converge and that the 
            behaviour of the system is highly unreliable. This could be due to numerical issues.
        """
        if t:
            tf = t
        else:
            tf = self.t1
            
        if D.abs(tf - self.t[-1]) < D.epsilon():
            if self.dense_output:
                self.compute_dense_output()
            return

        steps   = 0
        
        self.__fix_dt_dir(tf, self.t[-1])

        if D.abs(self.dt) > D.abs(tf - self.t[-1]):
            self.dt = D.abs(tf - self.t[-1])*0.5

        total_steps = int((tf-self.t[-1])/self.dt)
        
        if eta:
            tqdm_progress_bar = tqdm(total=total_steps)

        self.nfev = 0 if self.int_status == 1 else self.nfev
        dState  = D.zeros_like(self.y[-1])
        dTime   = D.zeros_like(self.t[-1])
        cState  = D.zeros_like(self.y[-1])
        cTime   = D.zeros_like(self.t[-1])
        self.__allocate_soln_space(total_steps)
        while self.dt != 0 and D.abs(self.t[-1]) < D.abs(tf * (1 - D.epsilon())):
            try:
                if abs(self.dt + self.t[-1]) > D.abs(tf):
                    self.dt = (tf - self.t[-1])
                try:
                    self.dt, (dTime, dState) = self.integrator(self.equ_rhs, self.t[-1], self.y[-1], self.consts, timestep=self.dt)
                except etypes.RecursionError:
                    print("Hit Recursion Limit. Will attempt to compute again with a smaller step-size. ",
                          "If this fails, either use a different rtol/atol or ",
                          "increase maximum recursion depth.", file=sys.stderr)
                    self.dt = 0.5 * self.dt
                    self.dt, (dTime, dState) = self.integrator(self.equ_rhs, self.t[-1], self.y[-1], self.consts, timestep=self.dt)
                except:
                    self.int_status = -1
                    raise
                
                if self.counter+1 >= len(self._y):
                    total_steps = int((tf-self._t[self.counter]-dTime)/self.dt) + 1
                    self.__allocate_soln_space(total_steps)
                
                #
                # Compensated Summation based on 
                # https://reference.wolfram.com/language/tutorial/NDSolveSPRK.html
                #
                
                dState = dState + cState
                dTime  = dTime  + cTime
                
                self._y[self.counter+1] = self._y[self.counter] + dState
                self._t[self.counter+1] = self._t[self.counter] + dTime 
                
                cState = (self._y[self.counter] - self._y[self.counter+1]) + dState
                cTime  = (self._t[self.counter] - self._t[self.counter+1]) + dTime
                
                self.counter += 1
                
                if eta:
                    tqdm_progress_bar.total = tqdm_progress_bar.n + int(abs(tf - self.t[-1]) / self.dt)
                    tqdm_progress_bar.desc  = f"{self.t[-1]:>10.2f} | {tf:.2f} | {self.dt:<10.2e}"
                    tqdm_progress_bar.update()
                steps += 1
                if callback is not None:
                    if isinstance(callback, (list, tuple)):
                        for i in callback:
                            i(self)
                    else:
                        callback(self)
                        
            except KeyboardInterrupt:
                self.int_status = -2
                if eta:
                    tqdm_progress_bar.close()
                self.__trim_soln_space()
                raise
            except:
                self.int_state = -3
                if eta:
                    tqdm_progress_bar.close()
                self.__trim_soln_space()
                raise

        self.__trim_soln_space()
        self.int_status = 1
        self.success = True
        if self.dense_output:
            self.compute_dense_output()

    def __repr__(self):
        return "\n".join([
            """{:>10}: {:<128}""".format("message", self.integration_status()),
            """{:>10}: {:<128}""".format("nfev", str(self.nfev)),
            """{:>10}: {:<128}""".format("sol", str(self.sol)),
            """{:>10}: {:<128}""".format("t", str(self.t)),
            """{:>10}: {:<128}""".format("y", str(self.y)),
        ])

    def __str__(self):
        """Prints the equations, initial conditions, final states, time limits and defined constants in the system."""
        print_str = "y({t0}) = {init_val}\ndy = {equation}\ny({t}) = {cur_val}\n"
        print_str = print_str.format(init_val=self.y[0], 
                                     t0=self.t0,
                                     equation=str(self.equ_rhs), 
                                     t=self.t[-1],
                                     cur_val=self.y[-1])
        if self.consts:
            print_str += "\nThe constants that have been defined for this system are: "
            print_str += "\n" + str(self.consts)
        
        print_str += "The time limits for this system are:\n"
        print_str += "t0 = {}, t1 = {}, t_current = {}, step_size = {}".format(self.t0, self.t1, self.t[-1], self.dt)
        
        return print_str
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return (self.t[index], self.y[index])
        elif isinstance(index, float):
            if self.dense_output and self.sol is not None:
                return (index, self.sol[index])
            else:
                nearest_idx = deutil.search_bisection(self.t, index)
                return (self.t[nearest_idx], self.y[nearest_idx])
            
    def __len__(self):
        return self.counter + 1