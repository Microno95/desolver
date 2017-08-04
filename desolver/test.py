import sympy as smp
import numpy as np
import re
import sys
import os
import time

class SanitizeInputs():

    _sanitization_regex = \
        r"(\.*\_*(builtins|import|class|(?<!(c|C))os|shutil|sys|time|dict|tuple|list|module|super|name|subclasses|base|lambda)\_*)"
    _compiled_regex = None

    def __new__(cls, string_to_be_sanitized):
        if SanitizeInputs._compiled_regex is None:
            SanitizeInputs._compiled_regex = re.compile(SanitizeInputs._sanitization_regex)
        if isinstance(string_to_be_sanitized, (str, bytes, bytearray)):
            return SanitizeInputs._compiled_regex.sub("", string_to_be_sanitized)
        else:
            return string_to_be_sanitized


def bisectroot(equn, n, h, m, vardict, low, high, cstring, iterlimit=None):
    """
    Uses the bisection method to find the zeros of the function defined in cstring.
    Designed to be used as a method to find the value of the next y_#.
    """
    import copy as cpy
    if iterlimit is None:
        iterlimit = 64  # Iteration limit for bisection method
    r = 0  # Track iteration count
    temp_vardict = cpy.deepcopy(vardict)
    temp_vardict.update({'t': vardict['t'] + m * h[0]})
    while np.amax(np.abs(np.subtract(high, low))) > 1e-14 and r < iterlimit:
        temp_vardict.update({'y_{}'.format(n): (low + high) * 0.5})
        if r > iterlimit:
            break
        c = eval(cstring)
        if (np.sum(c) >= 0 and np.sum(low) >= 0) or (np.sum(c) < 0 and np.sum(low) < 0):
            low = c
        else:
            high = c
        r += 1
    vardict.update({'y_{}'.format(n): vardict['y_{}'.format(n)] + (low + high) * 0.5 / m})


def extrap(x, xdata, ydata):
    """
    Richardson Extrapolation.

    Calculates the extrapolated values of a function evaluated at xdata
    with the values ydata.

    Required Arguments:
    x : The value(s) to extrapolate to (can be a numpy array)
    xdata, ydata : Values at which a function is evaluated, the result of that evaluation
    """
    coeff = []
    xlen = len(xdata)
    coeff.append([0, ydata[0]])
    for j in range(1, xlen):
        try:
            coeff.append([0, ydata[j]])
            for k in range(2, j + 1):
                if np.all([(i < 5e-14) for i in np.abs(coeff[-2][-1] - coeff[-1][-1])]):
                    raise StopIteration
                t1 = xdata[j] - xdata[j - k + 1]
                s1 = (x - xdata[j - k + 1])
                s2 = (x - xdata[j])
                p1 = s1 * coeff[j][k - 1] - s2 * coeff[j - 1][k - 1]
                p1 /= t1
                coeff[j].append(p1)
        except StopIteration:
            coeff.pop()
            break
    return coeff


def convertSuffix(value=3661, suffixes=[' d', ' h', ' m', ' s'], ratios=[24, 60, 60], delimiter=':'):
    """
    Converts a base value into a human readable format with the given suffixes and ratios.
    """
    tValue = value
    outputValues = []
    for i in ratios[::-1]:
        outputValues.append(int(tValue % i))
        tValue = (tValue - tValue % i) // i
    outputValues.append(int(tValue))
    return delimiter.join(["{:2d}{}".format(*i) for i in zip(outputValues[::-1], suffixes)])


def ConvertToIterable(item_to_be_converted, conversion_class, excepted_types=[]):
    if isinstance(item_to_be_converted, tuple(excepted_types)):
        if isinstance(item_to_be_converted, list) and (conversion_class is set or conversion_class is dict):
            ret_item = tuple(item_to_be_converted[:])
        else:
            ret_item = item_to_be_converted[:]
        return conversion_class([ret_item])
    else:
        try:
            iterable = iter(item_to_be_converted)
        except TypeError as e:
            if str(e).endswith("object is not iterable"):
                return conversion_class([item_to_be_converted])
        else:
            return conversion_class(item_to_be_converted)

class OdeSystem():
    """
    Class representing a system of Ordinary Differential Equations

    Stores system of equations, variables and constants
    """

    def __init__(self, diff_equations, _constants=()):
        self.n = len(diff_equations)

        self.equations = ConvertToIterable(diff_equations, conversion_class=list, excepted_types=[str])
        for i in range(self.n):
            if isinstance(self.equations[i], str):
                self.equations[i] = smp.sympify(SanitizeInputs(self.equations[i]))

        self.variables = set(smp.symbols("y_0:{}".format(len(self.equations))))
        self.time = smp.symbols("t")
        self.constants = set.union(*[expr.free_symbols for expr in self.equations]) - self.variables - {self.time}
        self.constants = {key.name: 0 for key in self.constants}

        self.lambdified_equations = [smp.lambdify(self.variables.union(self.constants, {self.time}), equ, "numpy", dummify=False) for equ in self.equations]

        self.equations = {smp.Symbol("y_{}".format(i)): self.equations[i] for i in range(self.n)}
        self.lambdified_equations = {smp.Symbol("y_{}".format(i)): self.lambdified_equations[i] for i in range(self.n)}

        self.constants.update(dict(_constants))
        self.constants = {key: SanitizeInputs(value) for key, value in self.constants.items()}

    def __call__(self, *args, **kwargs):
        return {key.name: equ(*args, **kwargs, **self.constants) for key, equ in self.lambdified_equations.items()}

class OdeIntegrator():
    raise_kbInt = False

    def __init__(self, diff_equations, _constants, eta=False, t_init=0, t_final=1, dt=1, initial_values=None):
        self.ode_equations = OdeSystem(diff_equations, _constants)
        initial_values = dict() if initial_values is None else \
            ConvertToIterable(initial_values, conversion_class=dict, excepted_types=[list, tuple])
        self.variable_states = {key.name: initial_values[key.name] if key in initial_values else 0 for key in self.ode_equations.variables}
        self.variable_states['t'] = t_init
        self.t_init = t_init
        self.t_final = t_final
        self.dt = dt
        self.eta = eta

    def integrate(self, t=None):
        """Integrates the system to a specified time.

        Keyword arguments:
        t: If t is specified, then the system will be integrated to time t.
           Otherwise the system will integrate to the specified final time.
           NOTE: t can be negative in order to integrate backwards in time, but use this with caution as this
                 functionality is slightly unstable."""


    def integrateOneStep(self):
        pass


if __name__ == "__main__":
    a = OdeSystem(['log(y_0 + 1)', 'y_0 * y_1', '-y_2 * y_1 + y_0','t*y_0/(y_1 + y_3 + 1)'])
    b = OdeIntegrator(['log(y_0 + 1)', 'y_0 * y_1', '-y_2 * y_1 + y_0','t*y_0/(y_1 + 1 + y_3)'], {})

    print(a.equations, a.variables, a.lambdified_equations)
    print(a(y_0=5, y_1=2, y_2=1, y_3=3, t=4))
    print(b.variable_states)
    args = b.variable_states
    args.update(a(y_0=5, y_1=2, y_2=1, y_3=3, t=4))
    print(a(**args))