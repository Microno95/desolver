"""
The MIT License (MIT)

Copyright (c) 2017 Microno95, Ekin Ozturk

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

import re
import numpy
import numpy.linalg
import sys
import time
import shutil
import types
from scipy.interpolate import CubicSpline

from tqdm.auto import tqdm

from . import integrationschemes as ischemes
from . import exceptiontypes as etypes

namespaceInitialised = False
available_methods = {}
methods_inv_order = {}
raise_KeyboardInterrupt = False

def init_module(raiseKBINT=True):
    global namespaceInitialised
    if not namespaceInitialised:
        global raise_KeyboardInterrupt
        raise_KeyboardInterrupt = raiseKBINT
        methods_ischemes = []
        # print(dir(ischemes))
        for a in dir(ischemes):
            try:
                if issubclass(ischemes.__dict__.get(a), ischemes.IntegratorTemplate):
                    methods_ischemes.append(ischemes.__dict__.get(a))
            except TypeError:
                pass
        # print(methods_ischemes)
        available_methods.update(dict([(func.__name__, func) for func in methods_ischemes if hasattr(func, "__alt_names__")] +
                                      [(alt_name, func) for func in methods_ischemes if hasattr(func, "__alt_names__") for alt_name in func.__alt_names__]))

        methods_inv_order.update({func: 1.0/func.__order__ for name, func in available_methods.items()})

        namespaceInitialised = True
    elif raiseKBINT:
        raise_KeyboardInterrupt = True
    else:
        pass

class DiffRHS:
    def __init__(self, rhs, equRepr=""):
        self.rhs = rhs
        self.equRepr = equRepr
    def __call__(self, t, y, **kwargs):
        return self.rhs(t, y, **kwargs)

    def __str__(self):
        return self.equRepr

    def __repr__(self):
        return "<DiffRHS({},{})>".format(repr(self.rhs), self.equRepr)

def rhs_prettifier(equRepr):
    def rhs_wrapper(rhs):
        return DiffRHS(rhs, equRepr)
    return rhs_wrapper

class OdeSystem:
    """Ordinary Differential Equation class. Designed to be used with a system of ordinary differential equations."""
    def __init__(self, equ_rhs, y0=None, n=(1,), t=(0, 1), savetraj=0, stpsz=1.0, eta=0, rtol=1e-6, atol=1e-6, constants=None):
        """Initialises the system to the parameters passed or to default values.

        Required arguments:
            equ_rhs: Specifies the right hand side of the system.
                     The calling signature of equ_rhs should be:
                     equ_rhs(t, y, **constants)
                     where **constants refers to keyword arguments that define
                     the constants of the system. These can be defined either
                     upon calling OdeSystem.__init__ using the constants
                     keyword argument or using the OdeSystem add_constants
                     method.
                     NOTE:
                        To make the output prettier, you can decorate the rhs
                        function with a @rhs_prettifier("Equation representation")
                        call where "Equation Representation" is a text representation
                        of your equation.

        Keyword arguments:
            y0: Specifies the initial state of the system. This is set to
                numpy.zeros(n) by default.
            n: Specifies the dimensions of the system in the form of a tuple.
               Can be arbitrary as long as the values are integral. Uses numpy convention so
               a scalar is (1,), a vector is (1,3), a 2x2 matrix is (1,2,2),
               an (M,T) tensor is (1,n_1,...,n_M,k_1,...,k_T) as expected.
               A numpy array of shape n will be passed to the integration routines,
               thus equ_rhs should be able to take this as input.
            t: A tuple of the form (initial time, final time) aka the integration limits.
            savetraj: Set to True or False to specify whether or not the trajectory of the
                      integration should be recorded.
            stpsz: Sets the step-size for the integration, choose a value that is slightly less than the highest frequency
                   changes in value of the solutions to the equations.
            eta: Set to True or False to specify whether or not the integration process should return an eta,
                 current progress and simple information regarding step-size and current time.
                 NOTE: This may slow the integration process down as the process of outputting
                       these values create overhead.
            relerr: Denotes the target relative global error. Useful for adaptive methods.

        Variable-length arguments:
            consts: Arbitrary set of keyword arguments that define the constants to be used in the system."""

        if len(t) != 2:
            raise etypes.LengthError("Two time bounds were required, only {} were given.".format(len(t)))
        if not hasattr(equ_rhs, "__call__"):
            raise TypeError("equ_rhs is not callable, please pass a callable object for the right hand side.")
        self.equ_rhs = equ_rhs
        self.rtol = rtol
        self.atol = atol
        self.consts = constants if constants is not None else dict()
        self.eta = eta
        self.dim = tuple(list(n))
        self.y = numpy.array(y0) if y0 is not None else numpy.zeros(self.dim)
        if not numpy.all(self.y.shape == self.dim):
            raise ValueError("initial y0 has the wrong shape, expected y0 with shape {} but got with shape {}".format(self.dim, self.y.shape))
        self.t = float(t[0])
        self.sample_times = [self.t]
        self.t0 = float(t[0])
        self.t1 = float(t[1])
        self.soln = [numpy.resize(self.y, self.dim)]
        self.traj = savetraj
        self.method_name = "RK45CK"
        self.integrator  = None
        if (stpsz < 0 < t[0] - t[1]) or (stpsz > 0 > t[0] - t[1]):
            self.dt = stpsz
        else:
            self.dt = -1 * stpsz
        self.staggered_mask = None
        self.dense_output = dense_output
        self.initialise_integrator()

    def set_velocity_vars(self, staggered_mask):
        """Sets the velocity variable mask for the symplectic integrators. Does nothing
        if integrator is not symplectic.
        """

        self.staggered_mask = staggered_mask

    def set_end_time(self, t):
        """Changes the final time for the integration of the ODE system

        Required arguments:
        t: Denotes the final time."""
        self.t1 = float(t)
        self.check_time_bounds()

    def get_end_time(self):
        """Returns the final time of the ODE system."""
        return self.t1

    def set_start_time(self, t):
        """Changes the initial time for the integration of the ODE system.

        Required arguments:
        t: Denotes the initial time."""
        self.t0 = float(t)
        self.check_time_bounds()

    def get_start_time(self):
        """Returns the initial time of the ODE system."""
        return self.t0

    def check_time_bounds(self):
        if not (abs(self.t0) < abs(self.t) < abs(self.t1)):
            self.t = self.t0

    def set_current_time(self, t):
        """Changes the current time for the integration of the ODE system.

        Required arguments:
        t: Denotes the current time"""
        self.t = float(t)

    def get_current_time(self):
        """Returns the current time of the ODE system"""
        return self.t

    def set_time(self, t=()):
        """Alternate interface for changing current, beginning and end times.

        Keyword arguments:
        t:  -- A length of 1 denotes changes to current time.
            -- A length of 2 denotes changes to beginning and end times in that order.
            -- A length of 3 denotes changes to all three times in order of current, beginning and end.
            -- A length larger than 3 will behave the same as above and ignore values beyond the 3rd index."""
        if len(t) == 1:
            ischemes.deutil.warning("You have passed a tuple that only contains one element, "
                    "this will be taken as the current time.")
            self.t = t[0]
        elif len(t) == 2:
            self.t0 = t[0]
            self.t1 = t[1]
        elif len(t) == 3:
            self.t = t[0]
            self.t0 = t[1]
            self.t1 = t[2]
        elif len(t) > 3:
            ischemes.deutil.warning("You have passed an array longer than 3 elements, "
                    "the first three will be taken as the principle values.")
            self.t = t[0]
            self.t0 = t[1]
            self.t1 = t[2]
        else:
            ischemes.deutil.warning("You have passed an array that is empty, this doesn't make sense.")

    def set_step_size(self, h):
        """Sets the step size that will be used for the integration.

        Required arguments:
        dt: Step size value. For systems that grow exponentially choose a smaller value, for oscillatory systems choose
            a value slightly less than the highest frequency of oscillation. If unsure, use an adaptive method in the
            list of available methods (view by calling availmethods()) followed by setmethod(), and finally call
            setrelerr() with the keyword argument auto_calc_dt set to True for an approximately good step size."""
        self.dt = h
        self.initialise_integrator()

    def get_step_size(self):
        """Returns the step size that will be attempted for the next integration step"""
        return self.dt

    def set_relative_error(self, relerr):
        """Sets the target relative error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive. These are all the symplectic integrators and the fixed order schemes.
        """
        self.relative_error_bound = relerr
        self.initialise_integrator()

    def set_rtol(self, new_rtol):
        """Sets the target relative error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive..
        """
        self.rtol = new_rtol

    def set_atol(self, new_atol):
        """Sets the target absolute error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive..
        """
        self.atol = new_atol

    def get_rtol(self):
        """Returns the target relative error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive..
        """
        return self.rtol

    def get_atol(self):
        """Returns the target absolute error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive..
        """
        return self.atol

    @staticmethod
    def available_methods(suppress_print=False):
        """Prints and then returns a dict of methods of integration that are available."""
        if not suppress_print:
            print(available_methods.keys())
        return available_methods

    def initialise_integrator(self):
        if available_methods[self.method_name].__adaptive__:
            if self.staggered_mask is not None and available_methods[self.method_name].__symplectic__:
                self.integrator = available_methods[self.method_name](self.dim, staggered_mask=self.staggered_mask, rtol=self.rtol, atol=self.atol)
            else:
                self.integrator = available_methods[self.method_name](self.dim, rtol=self.rtol, atol=self.atol)
        else:
            if self.staggered_mask is not None and available_methods[self.method_name].__symplectic__:
                self.integrator = available_methods[self.method_name](self.dim, staggered_mask=self.staggered_mask)
            else:
                self.integrator = available_methods[self.method_name](self.dim)

    def set_method(self, method, staggered_mask=None):
        """Sets the method of integration.

        Required arguments:
        method: String that denotes the key to one of the available methods in the dict() returned by availmethods()."""
        if method in available_methods.keys():
            self.method_name = method
            if staggered_mask is None and hasattr(self.integrator, "staggered_mask"):
                staggered_mask = self.integrator.staggered_mask
            self.staggered_mask = staggered_mask
            self.integrator = None
        else:
            raise ValueError("The method you selected does not exist in the list of available methods, \
                              call desolver.available_methods() to see what these are")
        self.initialise_integrator()

    def show_equations(self):
        """Prints the equations that have been entered for the system.
        Returns the equations themselves as a list of strings."""
        print("dy = {}".format(str(self.equ_rhs)))
        return str(self.equ_rhs)

    def show_system(self):
        """Prints the equations, initial conditions, final states, time limits and defined constants in the system."""
        print_str = "y({t0}) = {init_val}\ndy = {equation}\ny({t}) = {cur_val}\n"
        print(print_str.format(init_val=self.y, t0=self.t0,
                               equation=str(self.equ_rhs), t=self.t,
                               cur_val=self.soln[-1]))
        if self.consts:
            print("The constants that have been defined for this system are: ")
            print(self.consts)
        print("The time limits for this system are:\n "
              "t0 = {}, t1 = {}, t_current = {}, step_size = {}".format(self.t0, self.t1, self.t, self.dt))

    def add_constants(self, **additional_constants):
        """Takes an arbitrary list of keyword arguments to add to the list of available constants.

        Variable-length arguments:
        additional_constants: A dict containing constants and their corresponding values."""
        self.consts.update(additional_constants)

    def remove_constants(self, **constants_removal):
        """Takes an arbitrary list of keyword arguments to remove from the list of available constants.

        Variable-length arguments:
        additional_constants: A tuple or list containing the names of the constants to remove.
                              The names must be denoted by strings."""
        for i in constants_removal:
            if i in self.consts.keys():
                del self.consts[i]

    def set_dimensions(self, m=None):
        """Changes the dimensions of the system.

        Keyword arguments:
        m: Takes a tuple that describes the dimensions of the system. For example, to integrate 3d vectors one would
        pass (3,)."""
        if m is not None:
            if isinstance(m, float):
                raise ValueError('The dimension of a system cannot be a float')
            elif isinstance(m, int):
                self.dim = (m,)
            else:
                if any([not isinstance(m_elem, int) for m_elem in m]):
                    raise ValueError("The dimensions of a system cannot contain a float")
                self.dim = tuple([1] + list(m))
            self.y = numpy.resize(self.y, self.dim)
            self.reset()

        self.initialise_integrator()

    def set_dense_output(self, dense_output=True):
        """Sets self.dense_output to dense_output which determines if a CubicSpline
        fit is computed for the integration results.

        WARNING: If dense_output is changed from its original value,
                 the system will be reset.
        """
        if dense_output != self.dense_output:
            self.reset()
        self.dense_output = dense_output

    def reset(self):
        """Resets the system to the initial time."""
        self.t = self.t0
        self.soln = [self.y]
        self.sample_times = [self.t]

    def integrate(self, t=None, callback=None):
        """Integrates the system to a specified time.

        Keyword arguments:
        t: If t is specified, then the system will be integrated to time t.
           Otherwise the system will integrate to the specified final time.
           NOTE: t can be negative in order to integrate backwards in time, but use this with caution as this
                 functionality is slightly unstable."""

        if t:
            tf = t
        else:
            tf = self.t1

        steps = 0
        self.dt = self.dt

        if numpy.sign(tf - self.t) != numpy.sign(self.dt):
            self.dt = numpy.copysign(self.dt, tf - self.dt)

        if abs(self.dt) > abs(tf - self.t):
            self.dt = abs(tf - self.t)*0.5

        time_remaining = [0, 0]

        etaString = ''

        if self.eta:
            tqdm_progress_bar = tqdm(total=(tf-self.t)/self.dt)

        while self.dt != 0 and abs(self.t) < abs(tf * (1 - 4e-16)):
            try:
                if abs(self.dt + self.t) > abs(tf):
                    self.dt = (tf - self.t)
                try:
                    self.dt, (new_time, new_state) = self.integrator(self.equ_rhs, self.t, self.soln[-1], self.consts, timestep=self.dt)
                except etypes.RecursionError:
                    print("Hit Recursion Limit. Will attempt to compute again with a smaller step-size. "
                          "If this fails, either use a different relative error requirement or "
                          "increase maximum recursion depth. Can also occur if the initial value of all "
                          "variables is set to 0.")
                    self.dt = 0.5 * self.dt
                    self.dt, (new_time, new_state) = self.integrator(self.equ_rhs, self.t, self.soln[-1], self.consts, timestep=self.dt)
                except:
                    raise

                self.t = new_time

                if self.dense_output:
                    self.soln += [new_state]
                    self.sample_times.append(new_time)
                else:
                    self.soln = [new_state]
                    self.sample_times = [new_time]


                if self.eta:
                    tqdm_progress_bar.total = tqdm_progress_bar.n + int(abs(tf - self.t) / self.dt)
                    tqdm_progress_bar.update()
                steps += 1
                if callback is not None: callback(self)
            except KeyboardInterrupt:
                if raise_KeyboardInterrupt: raise
            except:
                raise

        self.t = self.sample_times[-1]
        if self.dense_output:
            self.spline_interp = CubicSpline(self.sample_times, self.soln, extrapolate=True)
