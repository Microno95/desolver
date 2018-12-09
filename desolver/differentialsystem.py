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
import sympy as smp
from sympy.parsing.sympy_parser import parse_expr

from tqdm.auto import tqdm

import desolver.integrationschemes as ischemes
import desolver.exceptiontypes as etypes

namespaceInitialised = False
available_methods = {}
methods_inv_order = {}
raise_KeyboardInterrupt = False

# This regex string will match any unacceptable arguments attempting to be passed to eval
#precautions_regex = r"([^a-z]*[^A-z]*)((\.*\_*)(builtins|class|(?<!(c|C))os|shutil|sys|time|dict|tuple|list|module|super|name|subclasses|base|lambda)(\_*\.*))([^a-z]*[^A-z]*)"
precautions_regex = "(?!)"

def init_module(raiseKBINT=True):
    global namespaceInitialised
    if not namespaceInitialised:
        global raise_KeyboardInterrupt
        raise_KeyboardInterrupt = raiseKBINT
        global precautions_regex
        precautions_regex = re.compile(precautions_regex)
        methods_ischemes = []
        for a in dir(ischemes):
            try:
                if issubclass(ischemes.__dict__.get(a), ischemes.integrator_template):
                    methods_ischemes.append(ischemes.__dict__.get(a))
            except:
                if isinstance(ischemes.__dict__.get(a), types.FunctionType):
                    methods_ischemes.append(ischemes.__dict__.get(a))
        
        print(methods_ischemes)
        available_methods.update(dict([(func.__name__, func) for func in methods_ischemes if hasattr(func, "__alt_names__")] +
                                        [(alt_name, func) for func in methods_ischemes if hasattr(func, "__alt_names__") for alt_name in func.__alt_names__]))

        methods_inv_order.update({func: 1.0/func.__order__ for name, func in available_methods.items()})

        namespaceInitialised = True
    elif raiseKBINT:
        raise_KeyboardInterrupt = True
    else:
        pass

class OdeSystem:
    """Ordinary Differential Equation class. Designed to be used with a system of ordinary differential equations."""
    def __init__(self, n=(1,), equ=tuple(tuple()), t=(0, 0), savetraj=0, stpsz=1.0, eta=0, relerr=4e-16, constants=None):
        """Initialises the system to the parameters passed or to default values.

        Keyword arguments:
            n: Specifies the dimensions of the system in the form of a tuple.
               Can be arbitrary as long as the values are integral. Uses numpy convention so
               a scalar is (1,), a vector is (1,3), a 2x2 matrix is (1,2,2),
               an (M,T) tensor is (1,n_1,...,n_M,k_1,...,k_T) as expected.
            equ: Specifies the list of differential equations and their initial conditions.
                 Use in the form of strings where t and y_{} are the variables.
                 The curly braces are to be replaced by values that range from 0 to k where
                 k = total_number_of_equations - 1.
                 ie. y_0 will be the first equation, y_1 the second, and so on.
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
            raise etypes.LengthError("Two time bounds were required, only {} were given!".format(len(t)))
        self.relative_error_bound = relerr
        self.consts = constants if constants is not None else dict()
        self.symbols = set(smp.symbols("t " + " ".join([k for k in self.consts])))
        self.variables = set()
        self.var2index = dict()
        self.index2var = dict()
        self.equRepr = []
        for i in range(len(equ)):
            try:
                try:
                    temp = parse_expr(precautions_regex.sub("LUBADUBDUB", str(equ[i][1])))
                except (AttributeError, TypeError) as e:
                    if isinstance(equ[i][0], smp.Expr):
                        temp = equ[i][0]
                    else:
                        temp = parse_expr("t")
                    raise e
                except:
                    raise ValueError("{} could not be converted to an appropriate sympy function!".format(equ[i][1]))
            except ValueError as e:
                raise e
            except:
                print(equ[i][1])
                raise
            self.equRepr.append(temp)
                
            if len(equ[i]) == 2:
                temp = smp.Symbol("y_{}".format(i))
            elif len(equ[i]) == 3:
                if isinstance(equ[i][0], str):
                    temp = smp.Symbol(equ[i][0])
                elif isinstance(smp.Symbol):
                    temp = equ[i][0]
                else:
                    raise TypeError("Integration variable must be an instance of str or sympy.Symbol! - Received {} of type {}".format(equ[i][0], type(equ[i][0])))
            else:
                raise LengthError("Wrong number of arguments for equation {}! - {}".format(i, equ[i]))
            self.variables.update({temp})
            self.var2index.update({repr(temp): i})
            self.index2var.update({i: (temp, repr(temp))})
            
        self.symbols.update(self.variables)
        self.eta = eta
        self.eqnum = len(equ)
        self.equ = [smp.lambdify(self.symbols, i, numpy, dummify=False) for i in self.equRepr]
        self.dim = tuple(list(n))
        self.y = [None for i in range(self.eqnum)]
        for i in range(self.eqnum):
            if len(equ[i]) == 1:
                self.y[i] = numpy.resize(0.0, self.dim)
            else:
                self.y[i] = numpy.resize(equ[i][-1], self.dim)
        self.t = float(t[0])
        self.sample_times = [self.t]
        self.t0 = float(t[0])
        self.t1 = float(t[1])
        self.soln = [[numpy.resize(value, self.dim)] for value in self.y]
        self.traj = savetraj
        self.integrator = ischemes.explicitrk4
        if (stpsz < 0 < t[0] - t[1]) or (stpsz > 0 > t[0] - t[1]):
            self.dt = stpsz
        else:
            self.dt = -1 * stpsz

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

    def get_relative_error(self):
        """Returns the target relative error used by the timestep autocalculator and the adaptive integration methods.
        Has no effect when the integration method is non-adaptive. These are all the symplectic integrators and the fixed order schemes.
        """
        return self.relative_error_bound

    @staticmethod
    def available_methods(suppress_print=False):
        """Prints and then returns a dict of methods of integration that are available."""
        if not suppress_print:
            print(available_methods.keys())
        return available_methods
    
    def initialise_integrator(self):
        if available_methods[self.method_name].__adaptive__:
            self.integrator = available_methods[self.method_name](self.dt, self.dim, list(self.var2index.keys()), self.var2index, self.relative_error_bound)
        else:
            self.integrator = available_methods[self.method_name](self.dt, self.dim, list(self.var2index.keys()), self.var2index)
    
    def set_method(self, method):
        """Sets the method of integration.

        Required arguments:
        method: String that denotes the key to one of the available methods in the dict() returned by availmethods()."""
        if method in available_methods.keys():
            self.method_name = method
        else:
            raise ValueError("The method you selected does not exist in the list of available methods, \
                              call desolver.available_methods() to see what these are")
        
        self.initialise_integrator()
        
    def get_method(self):
        """
        Returns the method used to integrate the system.
        """
        return repr(self.integrator)

    def show_equations(self):
        """Prints the equations that have been entered for the system.

        Returns the equations themselves as a list of strings."""
        for i in range(self.eqnum):
            print("d{} =".format(self.index2var[i][1]), self.equRepr[i])
        return self.equRepr

    def number_of_equations(self):
        """Prints then returns the number of equations in the system"""
        print(self.eqnum)
        return self.eqnum

    def initial_conditions(self):
        """Prints the initial conditions of the system"""
        for i in range(self.eqnum):
            print("{}({}) = {}".format(self.index2var[i][1], self.t0, self.y[i]))
        return self.y

    def final_conditions(self, p=1):
        """Prints the final state of the system.

        Identical to initial conditions if the system has not been integrated"""
        if p:
            for i in range(self.eqnum):
                print("{}({}) = {}".format(self.index2var[i][1], self.t1, self.soln[i][-1]))
        return self.soln

    def show_system(self):
        """Prints the equations, initial conditions, final states, time limits and defined constants in the system."""
        for i in range(self.eqnum):
            print_str = "Equation {idx}\n{diff_var}({t0}) = {init_val}\nd{diff_var} = {equation}\n{diff_var}({t}) = {cur_val}\n"
            print(print_str.format(idx=i, diff_var=self.index2var[i][1], init_val=self.y[i], t0=self.t0,
                                   equation=str(self.equRepr[i]), t=self.t,
                                   cur_val=self.soln[i][-1]))
        if self.consts:
            print("The constants that have been defined for this system are: ")
            print(self.consts)
        print("The time limits for this system are:\n "
              "t0 = {}, t1 = {}, t_current = {}, step_size = {}".format(self.t0, self.t1, self.t, self.dt))

    def add_constants(self, **additional_constants):
        """Takes an arbitrary list of keyword arguments to add to the list of available constants.

        Variable-length arguments:
        additional_constants: A dict containing constants and their corresponding values."""
        self.consts.update({k: numpy.resize(additional_constants[k], self.dim) for k in additional_constants})
        self.symbols.update(smp.symbols(" ".join([k for k in self.consts])))
        self.equ = [smp.lambdify(self.symbols, i, "numpy", dummify=False) for i in self.equRepr]
        self.initialise_integrator()

    def remove_constants(self, **constants_removal):
        """Takes an arbitrary list of keyword arguments to remove from the list of available constants.

        Variable-length arguments:
        additional_constants: A tuple or list containing the names of the constants to remove.
                              The names must be denoted by strings."""
        for i in constants_removal:
            if i in self.consts.keys():
                self.symbols.remove(self.consts[i])
                del self.consts[i]
                
        self.equ = [smp.lambdify(self.symbols, i, "numpy", dummify=False) for i in self.equRepr]
        self.initialise_integrator()

    def set_dimensions(self, m=None):
        """Changes the dimensions of the system.

        Keyword arguments:
        m: Takes a tuple that describes the dimensions of the system. For example, to integrate 3d vectors one would
        pass (3,1)."""
        if m is not None:
            if isinstance(m, float):
                raise ValueError('The dimension of a system cannot be a float')
            elif isinstance(m, int):
                self.dim = (1, m,)
            else:
                if any([not isinstance(m_elem, int) for m_elem in m]):
                    raise ValueError("The dimensions of a system cannot contain a float")
                self.dim = tuple([1] + list(m))
            self.y = [numpy.resize(i, m) for i in self.y]
            solntime = self.soln[-1]
            self.soln = [[numpy.resize(i, m)] for i in self.soln[:-1]]
            self.soln.append(solntime)
            
        self.initialise_integrator()

    def get_dimensions(self):
        """Returns the dimensions of the current system in the form of a tuple of ints.

        Follows numpy convention so numpy dimensions and OdeSystem dimensions are interchangeable.
        """
        return self.dim

    def record_trajectory(self, b=None):
        """Sets whether or not the trajectory of the system will be recorded.

        Keyword arguments:
        b: A boolean value that denotes if the trajectory should be recorded.
           1 - implies record; 0 - implies don't record.
           If b is None, then this will return the current state of
           whether or not the trajectory is to be recorded"""
        if b is None:
            return self.traj
        else:
            self.traj = b

    def get_trajectory(self, var_names=tuple()):
        if isinstance(var_names, tuple):
            if len(var_names) == 0:
                return self.soln
            elif len(var_names) > 0:
                ret_val = [None for i in var_names]
                for idx,i in enumerate(var_names):
                    if isinstance(i, str):
                        ret_val[idx] = self.soln[self.var2index[i]]
                    elif isinstance(i,smp.Symbol):
                        ret_val[idx] = self.soln[self.var2index[repr(i)]]
                    elif isinstance(i, int):
                        ret_val[idx] = self.soln[i]
                    else:
                        raise TypeError("Type of {} is not appropriate and cannot be interpreted as an integration variable!")
                return ret_val
        elif isinstance(var_names, str):
            return [self.soln[self.var2index[i]]]
        elif isinstance(var_names, int):
            return [self.soln[var_names]]
        else:
            raise TypeError("var_names should either be a tuple of variable names or a single variable name as a string!")

    def get_sample_times(self):
        return self.sample_times

    def reset(self, t=None):
        """Resets the system to a previous time.

        Keyword arguments:
        t: If specified after the system has recorded the trajectory of the system during integration, then this will
           revert to that time. Otherwise the reversion of the system will be to its state prior to integration."""
        if t is not None:
            if self.traj:
                k = numpy.array(self.soln[-1])
                ind = numpy.argmin(numpy.square(numpy.subtract(k, t)))
                for i, k in enumerate(self.soln):
                    self.soln[i] = list(numpy.delete(k, numpy.s_[ind + 1:], axis=0))
                self.t = t
            else:
                ischemes.deutil.warning('Trajectory has not been recorded for prior integration, cannot revert to t = {}\nPlease '
                        'call reset() and record trajectory by calling recordtraj() before integrating'.format(t))
        else:
            for i in range(self.eqnum):
                self.soln[i] = [self.y[i]]
            self.t = self.t0
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
        heff = [self.dt, self.dt]
        
        if numpy.sign(tf - self.t) != numpy.sign(self.dt):
            heff = numpy.copysign(heff, tf - self.dt)
            self.dt = heff[0]
            
        if abs(heff[0]) > abs(tf - self.t):
            heff = [abs(tf - self.t)*0.5 for i in heff]
            
        heff.append(tf)
        
        time_remaining = [0, 0]
        vardict = {'t': self.t}
        vardict.update(self.consts)
        
        etaString = ''
        
        if self.eta:
            tqdm_progress_bar = tqdm(total=(tf-self.t)/heff[0])
        
        while heff[0] != 0 and heff[1] != 0 and abs(self.t) < abs(tf * (1 - 4e-16)):
            print(vardict)
            try:
                heff[1] = heff[0]
                
                if abs(heff[0] + self.t) > abs(tf):
                    heff[0] = (tf - self.t)
                rerun_step = True
                try:
                    soln, vardict, heff[0] = self.integrator(self.equ, vardict, self.soln)
                except etypes.RecursionError:
                    print("Hit Recursion Limit. Will attempt to compute again with a smaller step-size. "
                          "If this fails, either use a different relative error requirement or "
                          "increase maximum recursion depth. Can also occur if the initial value of all "
                          "variables is set to 0.")
                    heff[1] = heff[0]
                    heff[0] = 0.5 * heff[0]
                    self.integrator.step_size = heff[0]
                    soln, vardict, heff[0] = self.integrator(self.equ, vardict, self.soln)
                except:
                    raise
                    
                self.t = self.t + heff[0]
                
                if self.traj:
                    self.sample_times.append(vardict['t'])
                else:
                    self.soln = [numpy.array([i[-1]]) for i in self.soln]
                    self.sample_times = [vardict['t']]
                    
                    
                if self.eta:
                    if heff[0] != heff[1]:
                        tqdm_progress_bar.total = tqdm_progress_bar.n + int(abs(tf - self.t) / heff[0])
                    tqdm_progress_bar.update()
                    
                steps += 1
                if callback is not None: callback(self)
            except KeyboardInterrupt:
                if raise_KeyboardInterrupt: raise
            except:
                raise
                
        self.t = self.sample_times[-1]
        self.dt = heff[0]
