import numpy
import sys
import time as tm

safe_dict = {}
available_methods = {}
methods_inv_order = {}


# noinspection PyUnusedLocal
def bisectroot(equn, n, h, m, vardict, low, high, cstring, iterlimit=None):
    """
    Uses the bisection method to find the zeros of the function defined in cstring.
    Designed to be used as a method to find the value of the next y_#.
    """
    import copy as cpy
    if iterlimit is None:
        iterlimit = 64
    r = 0
    temp_vardict = cpy.deepcopy(vardict)
    temp_vardict.update({'t': vardict['t'] + m * h[0]})
    while numpy.sum(numpy.abs(numpy.subtract(high, low))) > 1e-14 and r < iterlimit:
        temp_vardict.update({'y_{}'.format(n): (low + high) * 0.5})
        if r > iterlimit:
            break
        c = eval(cstring)
        if (c >= 0 and low >= 0) or (c < 0 and low < 0):
            low = c
        else:
            high = c
        r += 1
    vardict.update({'y_{}'.format(n): vardict['y_{}'.format(n)] + (low + high) * 0.5 / m})


def extrap(x, xdata, ydata):
    coeff = []
    xlen = len(xdata)
    coeff.append([0, ydata[0]])
    for j in range(1, xlen):
        try:
            coeff.append([0, ydata[j]])
            for k in range(2, j + 1):
                if numpy.all([(i < 5e-14) for i in numpy.abs(coeff[-2][-1] - coeff[-1][-1])]):
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


def seval(string, **kwargs):
    """
    Safe eval() functions.
    Evaluates string within a namespace that excludes builtins and is limited to
    those defined in **kwargs if **kwargs is supplied.
    """
    safeglobals = {"__builtins__": None}
    safeglobals.update(kwargs)
    return eval(string, safeglobals, safe_dict)


def explicitrk4(ode, vardict, soln, h, relerr):
    """
    Implementation of the Explicit Runge-Kutta 4 method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    eqnum = len(ode)
    dim = [eqnum, 4]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if numpy.iscomplexobj(soln[0]):
        aux = numpy.resize([0. + 0j], dim)
    else:
        aux = numpy.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][0] * 0.5})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][1] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][1] * 0.5})
    for vari in range(eqnum):
        aux[vari][2] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][2]})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][3] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][3]})
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): (soln[vari][-1] +
                                              (aux[vari][0] + aux[vari][1] * 2 + aux[vari][2] * 2 + aux[vari][3]) / 6)})
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))


def explicitrk45ck(ode, vardict, soln, h, relerr, tol=0.5):
    """
    Implementation of the Explicit Runge-Kutta-Fehlberg method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    eqnum = len(ode)
    dim = [eqnum, 6]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    t_initial = vardict['t']
    if numpy.iscomplexobj(soln[0]):
        aux = numpy.resize([0. + 0j], dim)
    else:
        aux = numpy.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][0] / 5})
    vardict.update({'t': t_initial + h[0] / 5})
    for vari in range(eqnum):
        aux[vari][1] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + 3.0 * aux[vari][0] / 40 + 9.0 * aux[vari][1] / 40})
    vardict.update({'t': t_initial + 3 * h[0] / 10})
    for vari in range(eqnum):
        aux[vari][2] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + (3.0 * aux[vari][0] - 9.0 * aux[vari][1] +
                                                               12.0 * aux[vari][2]) / 10})
    vardict.update({'t': t_initial + 3 * h[0] / 5})
    for vari in range(eqnum):
        aux[vari][3] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): (soln[vari][-1] - 11.0 * aux[vari][0] / 54 - 5.0 * aux[vari][1] / 2 -
                                              70.0 * aux[vari][2] / 27 + 35.0 * aux[vari][3] / 27)})
    vardict.update({'t': t_initial + h[0]})
    for vari in range(eqnum):
        aux[vari][4] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): (soln[vari][-1] +
                                              1631.0 * aux[vari][0] / 55296 + 175.0 * aux[vari][1] / 512 +
                                              575.0 * aux[vari][2] / 13824 + 44275.0 * aux[vari][3] / 110592 +
                                              253.0 * aux[vari][4] / 4096)})
    vardict.update({'t': t_initial + 7.0 * h[0] / 8})
    for vari in range(eqnum):
        aux[vari][5] = numpy.resize(seval(ode[vari], **vardict) * h[0], dim)
    coeff = []
    for vari in range(eqnum):
        coeff.append([])
        coeff[vari].append(soln[vari][-1] + 37.0 * aux[vari][0] / 378 + 250.0 * aux[vari][2] / 621 +
                           125.0 * aux[vari][3] / 594 + 512.0 * aux[vari][5] / 1771)
        coeff[vari].append(soln[vari][-1] + 2825.0 * aux[vari][0] / 27648 + 18575.0 * aux[vari][2] / 48384 +
                           13525.0 * aux[vari][3] / 55296 + 277.0 * aux[vari][4] / 14336 + aux[vari][5] / 4.0)

    error_coeff_array = [[37.0 / 378 - 2825.0 / 27648], [0], [250.0 / 621 - 18575.0 / 48384],
                         [125.0 / 594 - 13525.0 / 55296], [-277.0 / 14336],
                         [512.0 / 1771 - 1.0 / 4.0]]
    error_coeff_array = [numpy.resize(i, dim) for i in error_coeff_array]
    est = [numpy.abs(numpy.sum(aux[vari] * error_coeff_array)) for vari in range(eqnum)]
    delta = numpy.abs(numpy.ravel([coeff[vari][0] + coeff[vari][1] for vari in range(eqnum)]))
    delta = numpy.sum(delta) / len(delta)
    est = numpy.sum(est) / len(est)
    if est != 0:
        h[1] = h[0]
        if est > delta * relerr:
            corr = h[0] * tol * (delta * relerr / est) ** (1.0 / 5.0)
        else:
            corr = h[0] * tol * (delta * relerr / est) ** (1.0 / 4.0)
        if abs(corr * 1.05 + t_initial) < abs(h[2]):
            if corr != 0:
                h[0] = corr
        else:
            h[0] = abs(h[2] - t_initial) / 2.0
    if est > delta * relerr:
        for vari in range(eqnum):
            vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        vardict.update({'t': t_initial})
        explicitrk45ck(ode, vardict, soln, h, relerr)
    else:
        for vari in range(eqnum):
            vardict.update({'y_{}'.format(vari): coeff[vari][1]})
            vardict.update({'t': t_initial + h[1]})
            pt = soln[vari]
            kt = numpy.array([coeff[vari][1]])
            soln[vari] = numpy.concatenate((pt, kt))


def explicitmidpoint(ode, vardict, soln, h, relerr):
    """
    Implementation of the Explicit Midpoint method.
    """
    eqnum = len(ode)
    dim = [eqnum, 2]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if numpy.iscomplexobj(soln[0]):
        aux = numpy.resize([0. + 0j], dim)
    else:
        aux = numpy.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize([seval(ode[vari], **vardict) * h[0] + soln[vari][-1]], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize([seval(ode[vari], **vardict)], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): numpy.array(soln[vari][-1] + h[0] * aux[vari][0])})
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})


def implicitmidpoint(ode, vardict, soln, h, relerr):
    """
    Implementation of the Implicit Midpoint method.
    """
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        bisectroot(ode[vari], vari, h, 0.5, vardict, - soln[vari][-1] - h[0] * seval(ode[vari], **vardict),
                   soln[vari][-1] + h[0] * seval(ode[vari], **vardict),
                   cstring="temp_vardict['y_{}'.format(n)] - h[0] * 0.5 * (seval(equn, **temp_vardict) "
                           "+ seval(equn, **vardict)) - vardict['y_{}'.format(n)]")
    for vari in range(eqnum):
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def heuns(ode, vardict, soln, h, relerr):
    """
    Implementation of Heun's method.
    """
    eqnum = len(ode)
    dim = [eqnum, 2]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if numpy.iscomplexobj(soln[0]):
        aux = numpy.resize([0. + 0j], dim)
    else:
        aux = numpy.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0] * h[0] + soln[vari][-1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][1] = numpy.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * (aux[vari][0] + aux[vari][1]) * 0.5})
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))


def backeuler(ode, vardict, soln, h, relerr):
    """
    Implementation of the Implicit/Backward Euler method.
    """
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        bisectroot(ode[vari], vari, h, 1.0, vardict, - soln[vari][-1] - h[0] * seval(ode[vari], **vardict),
                   soln[vari][-1] + h[0] * seval(ode[vari], **vardict),
                   cstring="temp_vardict['y_{}'.format(n)] - h[0] * seval(equn, **temp_vardict) "
                           "- vardict['y_{}'.format(n)]")
    for vari in range(eqnum):
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def foreuler(ode, vardict, soln, h, relerr):
    """
    Implementation of the Explicit/Forward Euler method.
    """
    eqnum = len(ode)
    dim = [eqnum, 1]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if numpy.iscomplexobj(soln[0]):
        aux = numpy.resize([0. + 0j], dim)
    else:
        aux = numpy.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize(seval(ode[vari], **vardict) * h[0] + soln[vari][-1], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def eulertrap(ode, vardict, soln, h, relerr):
    """
    Implementation of the Euler-Trapezoidal method.
    """
    eqnum = len(ode)
    dim = [eqnum, 3]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if numpy.iscomplexobj(soln[0]):
        aux = numpy.resize([0. + 0j], dim)
    else:
        aux = numpy.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        aux[vari][1] = numpy.resize(seval(ode[vari], **vardict) * h[0] + soln[vari][-1], dim)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): aux[vari][1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][2] = numpy.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * (aux[vari][0] + aux[vari][2])})
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))


def adaptiveheuneuler(ode, vardict, soln, h, relerr, tol=0.9):
    """
    Implementation of the Adaptive Heun-Euler method.
    """
    eqnum = len(ode)
    dim = [eqnum, 2]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if numpy.iscomplexobj(soln[0]):
        aux = numpy.resize([0. + 0j], dim)
    else:
        aux = numpy.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0] * h[0] + soln[vari][-1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][1] = numpy.resize(seval(ode[vari], **vardict), dim)
    err = [numpy.subtract(aux[vari][0] * h[0], (aux[vari][0] + aux[vari][1]) * 0.5 * h[0]) for vari in range(eqnum)]
    err = numpy.abs([numpy.abs(i) for i in err])
    err = numpy.amax(err)
    err *= ((h[2] - vardict['t'] + h[0]) / h[0])
    if numpy.any([err >= relerr]):
        vardict.update({'t': vardict['t'] - h[0]})
        if err != 0:
            h[0] *= tol * (relerr / err) ** (1.0 / 2.0)
        adaptiveheuneuler(ode, vardict, soln, h, relerr)
    else:
        if numpy.all([err ** 2.0 < relerr]) and err != 0:
            h[0] *= (relerr / err) ** (1.0 / 3.0)
            h[0] /= tol
        for vari in range(eqnum):
            vardict.update({"y_{}".format(vari): soln[vari][-1] + (aux[vari][0] + aux[vari][1]) * 0.5 * h[0]})
            pt = soln[vari]
            kt = numpy.array([vardict['y_{}'.format(vari)]])
            soln[vari] = numpy.concatenate((pt, kt))


def sympforeuler(ode, vardict, soln, h, relerr):
    """
    Implementation of the Symplectic Euler method.
    """
    eqnum = len(ode)
    dim = [eqnum, 1]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if numpy.iscomplexobj(soln[0]):
        aux = numpy.resize([0. + 0j], dim)
    else:
        aux = numpy.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = numpy.resize((seval(ode[vari], **vardict) * h[0] + soln[vari][-1]), dim)
        if vari % 2 == 0:
            vardict.update({"y_{}".format(vari): aux[vari][0]})
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def init_namespace():
    if len(safe_dict) == 0:
        import numpy
        safe_list = ['arccos', 'arcsin', 'arctan', 'arctan2', 'ceil', 'cos', 'cosh', 'degrees', 'e', 'exp', 'abs',
                     'fabs', 'floor', 'fmod', 'frexp', 'hypot', 'ldexp', 'log', 'log10', 'modf', 'pi', 'power',
                     'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'dot', 'vdot', 'outer', 'matmul',
                     'tensordot', 'inner', 'trace']
        global safe_dict
        safe_dict = {}

        for k in safe_list:
            safe_dict.update({'{}'.format(k): getattr(locals().get("numpy"), k)})
    else:
        pass
    if len(available_methods) == 0 or len(methods_inv_order) == 0:
        global available_methods
        global methods_inv_order
        available_methods = {"Explicit Runge-Kutta 4": explicitrk4, "Explicit Midpoint": explicitmidpoint,
                             "Symplectic Forward Euler": sympforeuler, "Adaptive Heun-Euler": adaptiveheuneuler,
                             "Heun's": heuns, "Backward Euler": backeuler, "Euler-Trapezoidal": eulertrap,
                             "Predictor-Corrector Euler": eulertrap, "Implicit Midpoint": implicitmidpoint,
                             "Forward Euler": foreuler, "Adaptive Runge-Kutta-Cash-Karp": explicitrk45ck}
        methods_inv_order = {"Explicit Runge-Kutta 4": 1.0/5.0, "Explicit Midpoint": 1.0/2.0,
                             "Symplectic Forward Euler": 1.0, "Adaptive Heun-Euler": 1.0/3.0,
                             "Heun's": 1.0/2.0, "Backward Euler": 1.0, "Euler-Trapezoidal": 1.0/3.0,
                             "Predictor-Corrector Euler": 1.0/3.0, "Implicit Midpoint": 1.0,
                             "Forward Euler": 1.0, "Adaptive Runge-Kutta-Cash-Karp": 1.0/5.0}
    else:
        pass


def warning(message):
    print(message)


class VariableMissing(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class LengthError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class OdeSystem:
    """Ordinary Differential Equation class. Designed to be used with a system of ordinary differential equations."""
    def __init__(self, n=(1,), equ=(), y_i=(), t=(0, 0), savetraj=0, stpsz=1.0, eta=0, relerr=4e-16, **consts):
        """Initialises the system to the parameters passed or to default values.

        Keyword arguments:
        n: Specifies the dimensions of the system in the form of a tuple.
           Can be arbitrary as long as the values are integral.
        equ: Specifies the list of differential equations. Use in the form of strings where t and y_{} are the variables.
             The curly braces are to be replaced by values that range from 0 to k where
             k = total_number_of_equations - 1.
             NOTE: y_0 will be the first equation, y_1 the second, and so on.
        y_i: Specifies the initial conditions of each equation in the order of equations that are passed.
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
        if len(equ) > len(y_i):
            raise LengthError("There are more equations than initial conditions!")
        elif len(equ) < len(y_i):
            warning("There are more initial conditions than equations!")
        elif len(t) != 2:
            raise LengthError("Two time bounds were required, only {} were given!".format(len(t)))
        for k, i in enumerate(equ):
            if 't' not in i and 'y_' not in i:
                warning("Equation {} has no variables".format(k))
        init_namespace()
        self.relative_error_bound = relerr
        self.equ = list(equ)
        self.y = [numpy.resize(i, n) for i in y_i]
        self.dim = tuple([1] + list(n))
        self.t = float(t[0])
        self.t0 = float(t[0])
        self.t1 = float(t[1])
        self.soln = [[numpy.resize(i, n)] for i in self.y]
        self.soln.append([t[0]])
        self.consts = consts
        for k in self.consts:
            self.consts.update({k: numpy.resize(self.consts[k], n)})
        self.traj = savetraj
        self.method = "Explicit Runge-Kutta 4"
        if (stpsz < 0 < t[0] - t[1]) or (stpsz > 0 > t[0] - t[1]):
            self.dt = -1 * stpsz
        else:
            self.dt = stpsz
        self.eta = eta
        self.eqnum = len(equ)

    def chgendtime(self, t):
        """Changes the final time for the integration of the ODE system

        Required arguments:
        t: Denotes the final time."""
        self.t1 = float(t)
        if not (abs(self.t0) < abs(self.t) < abs(self.t1)):
            self.t = self.t0

    def chgbegtime(self, t):
        """Changes the initial time for the integration of the ODE system.

        Required arguments:
        t: Denotes the initial time."""
        self.t0 = float(t)
        if not (abs(self.t0) < abs(self.t) < abs(self.t1)):
            self.t = self.t0

    def chgcurtime(self, t):
        """Changes the current time for the integration of the ODE system.

        Required arguments:
        t: Denotes the current time"""
        self.t = float(t)

    def chgtime(self, t=()):
        """Alternate interface for changing current, beginning and end times.

        Keyword arguments:
        t:  -- A length of 1 denotes changes to current time.
            -- A length of 2 denotes changes to beginning and end times in that order.
            -- A length of 3 denotes changes to all three times in order of current, beginning and end.
            -- A length larger than 3 will behave the same as above and ignore values beyond the 3rd index."""
        if len(t) == 1:
            warning("You have passed a tuple that only contains one element, "
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
            warning("You have passed an array longer than 3 elements, "
                    "the first three will be taken as the principle values.")
            self.t = t[0]
            self.t0 = t[1]
            self.t1 = t[2]
        else:
            warning("You have passed an array that is empty, this doesn't make sense.")

    def setstepsize(self, h):
        """Sets the step size that will be used for the integration.

        Required arguments:
        dt: Step size value. For systems that grow exponentially choose a smaller value, for oscillatory systems choose
            a value slightly less than the highest frequency of oscillation. If unsure, use an adaptive method in the
            list of available methods (view by calling availmethods()) followed by setmethod(), and finally call
            setrelerr() with the keyword argument auto_calc_dt set to True for an approximately good step size."""
        self.dt = h

    def setrelerr(self, relative_err, auto_calc_dt=0):
        """Sets the value for target relative global error, especially useful for adaptive methods.

        Required arguments:
        relative_err: Generally a float less than or equal to 1, do NOT set to 0 as that is an impossible goal
                      and will cause an infinite loop in adaptive methods.

        Keyword arguments:
        auto_calc_dt: if set to true, will use knowledge of the order of the method chosen in order to estimate
                      a good step size.
                      NOTE: This will not necessarily be the optimal step size, but will suffice in most cases."""
        self.relative_error_bound = relative_err
        if auto_calc_dt:
            alt_h = float(self.t1 - self.t) * (self.relative_error_bound ** methods_inv_order[self.method])
            if alt_h != 0:
                self.dt = alt_h
                print('Time step, dt, set to: {:.4e}'.format(self.dt))
            if relative_err <= 5e-16:
                print('This choice of relative error has been observed to cause issues with adaptive algorithms' +
                      ' and may lead to instability.\nIt is advised that you manually set the time-step to an' +
                      ' appropriate value and the relative error to 1e-15 or greater.\n\n')

    @staticmethod
    def availmethods():
        """Prints and then returns a dict of methods of integration that are available."""
        print(available_methods.keys())
        return available_methods

    def setmethod(self, method):
        """Sets the method of integration.

        Required arguments:
        method: String that denotes the key to one of the available methods in the dict() returned by availmethods()."""
        if method in available_methods.keys():
            self.method = method
        else:
            print("The method you selected does not exist in the list of available methods, "
                  "call availmethods() to see what these are")

    def addequ(self, eq, ic):
        """Adds an equation to an already defined system.

        Required arguments:
        eq: A list of strings that use y_{} and t as the integration variables where the curly braces should be
            replaced with the index of the equation being referenced starting from 0 as the first equation.
            NOTE: Referencing an equation that does not exist, ie. y_10 for a system with 10 equations will cause
                  the integrator to fail.
        ic: A list of values for each eq in order of the equations entered. If the dimension of the system is set
            to be larger than scalars then an initial condition with fewer specified values will be extended to the
            necessary dimensions. This may cause unexpected results and it is better to specify all the coefficients."""
        if eq and ic:
            for equation, icond in zip(eq, ic):
                self.equ.append(equation)
                self.y.append(numpy.resize(icond, self.dim))
            solntime = self.soln[-1]
            self.soln = [[numpy.resize([i], self.dim)] for i in self.y]
            self.soln.append(solntime)
            self.eqnum += len(eq)
            self.t = self.t0

    def delequ(self, indices):
        """Removes (an) equation(s) at (a) given ind(ex)(ices) along with its corresponding initial values.

        Required arguments:
        indices: A list of integers that denote the equations to remove from the system. Will throw an error if
                 there are more equations to be removed or if there is an index specified that exceeds the
                 number of equations that exist."""
        if len(indices) > self.eqnum:
            raise LengthError("You've specified the removal of more equations than there exists!")
        for i in indices:
            self.y.pop(i)
            self.soln.pop(i)
            self.equ.pop(i)
        self.eqnum -= len(indices)

    def showequ(self):
        """Prints the equations that have been entered for the system.

        Returns the equations themselves as a list of strings."""
        for i in range(self.eqnum):
            print("dy_{} = ".format(i) + self.equ[i])
        return self.equ

    def numequ(self):
        """Prints then returns the number of equations in the system"""
        print(self.eqnum)
        return self.eqnum

    def initcond(self):
        """Prints the initial conditions of the system"""
        for i in range(self.eqnum):
            print("y_{}({}) = {}".format(i, self.t0, self.y[i]))
        return self.y

    def finalcond(self, p=1):
        """Prints the final state of the system.

        Identical to initial conditions if the system has not been integrated"""
        if p:
            for i in range(self.eqnum):
                print("y_{}({}) = {}".format(i, self.t1, self.soln[i][-1]))
        return self.soln

    def showsys(self):
        """Prints the equations, initial conditions, final states, time limits and defined constants in the system."""
        for i in range(self.eqnum):
            print("Equation {}\ny_{}({}) = {}\ndy_{} = {}\ny_{}({}) = {}\n".format(i, i, self.t0, self.y[i], i,
                                                                                   self.equ[i], i, self.t,
                                                                                   self.soln[i][-1]))
        if self.consts:
            print("The constants that have been defined for this system are: ")
            print(self.consts)
        print("The time limits for this system are:\n "
              "t0 = {}, t1 = {}, t_current = {}, step_size = {}".format(self.t0, self.t1, self.t, self.dt))

    def addconsts(self, **additional_constants):
        """Takes an arbitrary list of keyword arguments to add to the list of available constants.

        Variable-length arguments:
        additional_constants: A dict containing constants and their corresponding values."""
        self.consts.update({k: numpy.resize(additional_constants[k], self.dim) for k in additional_constants})

    def remconsts(self, **constants_removal):
        """Takes an arbitrary list of keyword arguments to remove from the list of available constants.

        Variable-length arguments:
        additional_constants: A tuple or list containing the names of the constants to remove.
                              The names must be denoted by strings."""
        for i in constants_removal:
            if i in self.consts.keys():
                del self.consts[i]

    def chgdim(self, m=None):
        """Changes the dimensions of the system.

        Keyword arguments:
        m: Takes a tuple that describes the dimensions of the system. For example, to integrate 3d vectors one would
        pass (3,3)."""
        if m is not None:
            if isinstance(m, float):
                raise ValueError('The dimension of a system cannot be a float')
            elif isinstance(m, int):
                self.dim = (1, m,)
            else:
                self.dim = tuple([1] + list(m))
            self.y = [numpy.resize(i, m) for i in self.y]
            solntime = self.soln[-1]
            self.soln = [[numpy.resize(i, m)] for i in self.soln[:-1]]
            self.soln.append(solntime)

    def recordtraj(self, b=None):
        """Sets whether or not the trajectory of the system will be recorded.

        Keyword arguments:
        b: A boolean value that denotes if the trajectory should be recorded.
           1 - implies record; 0 - implies don't record. If b is None, then this will invert the sate from recording
           trajectories to not recording trajectories and vice versa."""
        if b is None:
            self.traj = (not self.traj)
        else:
            self.traj = b

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
                warning('Trajectory has not been recorded for prior integration, cannot revert to t = {}\nPlease '
                        'call reset() and record trajectory by calling recordtraj() before integrating'.format(t))
        else:
            for i in range(self.eqnum + 1):
                if i < self.eqnum:
                    self.soln[i] = [self.y[i]]
                else:
                    self.soln[i] = [0]
            self.t = 0

    def integrate(self, t=None):
        """Integrates the system to a specified time.

        Keyword arguments:
        t: If t is specified, then the system will be integrated to time t.
           Otherwise the system will integrate to the specified final time.
           NOTE: t can be negative in order to integrate backwards in time, but use this with caution as this
                 functionality is slightly unstable."""
        method = self.method
        if method is None:
            method = available_methods["Explicit Runge-Kutta 4"]
        else:
            try:
                method = available_methods[method]
            except KeyError:
                print("Method not available, defaulting to RK4 Method")
                method = available_methods["Explicit Runge-Kutta 4"]
        eta = self.eta
        heff = [self.dt, self.dt]
        if t:
            tf = t
        else:
            tf = self.t1
        steps = 0
        if tf - self.t < 0:
            heff = [-1.0 * abs(i) for i in heff]
        while abs(heff[0]) > abs(tf - self.t):
            heff = [i * 0.5 for i in heff]
        heff.append(tf)
        time_remaining = [0, 0]
        soln = self.soln
        vardict = {'t': self.t}
        vardict.update(self.consts)
        while heff[0] != 0 and abs(self.t) < abs(tf * (1 - 4e-16)):
            try:
                if heff[0] != heff[1]:
                    heff[1] = heff[0]
                if eta:
                    time_remaining[0] = tm.perf_counter()
                if abs(heff[0] * 1.03125 + self.t) - abs(tf) > abs(tf) * 4e-16:
                    heff[0] = (tf - self.t)
                elif heff[1] == 0 and heff[0] == 0:
                    break
                method(self.equ, vardict, soln, heff, self.relative_error_bound)
                if heff[0] == 0:
                    heff[0] = (tf - self.t) * 0.5
                self.t = vardict['t']
                if self.traj:
                    soln[-1].append(vardict['t'])
                else:
                    soln = [[i[-1]] for i in soln[:-1]]
                    soln.append([vardict['t']])
                if eta:
                    temp_time = 0.4 * time_remaining[1] + (((tf - self.t) / heff[0]) *
                                                           0.6 * (tm.perf_counter() - time_remaining[0]))
                    if temp_time != 0 and numpy.abs(time_remaining[1]/temp_time - 1) > 0.2:
                        time_remaining[1] = temp_time
                    sys.stdout.flush()
                    print('\r', end='')
                    print(
                        "{:.3f}% ----- approx. ETA: {} -- Current Time and Step Size: {:.4e} and {:.4e}".format(
                            round(100 - abs(tf - self.t) * 100 / abs(tf - self.t0), ndigits=3),
                            "{} minutes and {} seconds".format(
                                int(time_remaining[1] / 60.),
                                int(time_remaining[1] - int(
                                    time_remaining[1] / 60) * 60)), self.t, heff[0]), end='')
                steps += 1
            except KeyboardInterrupt:
                break
        if eta:
            sys.stdout.flush()
            print("\r100%")
        else:
            print("100%")
        self.soln = soln
        self.t = soln[-1][-1]
        self.dt = heff[0]
