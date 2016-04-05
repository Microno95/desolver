import numpy


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


def explicitrk4(ode, vardict, soln, h):
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
        vardict.update({'y_{}'.format(vari): soln[vari][-1] +
                       (aux[vari][0] + aux[vari][1] * 2 + aux[vari][2] * 2 + aux[vari][3]) / 6})
        pt = soln[vari]
        kt = numpy.array([vardict['y_{}'.format(vari)]])
        soln[vari] = numpy.concatenate((pt, kt))


def explicitmidpoint(ode, vardict, soln, h):
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


def implicitmidpoint(ode, vardict, soln, h):
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


def heuns(ode, vardict, soln, h):
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


def backeuler(ode, vardict, soln, h):
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


def foreuler(ode, vardict, soln, h):
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


def eulertrap(ode, vardict, soln, h):
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


def adaptiveheuneuler(ode, vardict, soln, h, tol=5e-16):
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
    err = 0.
    for vari in range(eqnum):
        temp_error = numpy.array(
            numpy.absolute(numpy.subtract(aux[vari][0] * h[0], (aux[vari][0] + aux[vari][1]) * 0.5 * h[0])))
        if numpy.any([temp_error > err]):
            err = temp_error
    if numpy.any([err > tol]):
        h[0] /= 2.0
        adaptiveheuneuler(ode, vardict, soln, h)
    else:
        if numpy.all([2 * err < tol]):
            h[0] *= 2
        for vari in range(eqnum):
            vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * aux[vari][0]})
            pt = soln[vari]
            kt = numpy.array([vardict['y_{}'.format(vari)]])
            soln[vari] = numpy.concatenate((pt, kt))


def sympforeuler(ode, vardict, soln, h):
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
    if 'safe_dict' not in globals():
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
    if 'available_methods' not in globals():
        global available_methods
        available_methods = {"Explicit Runge-Kutta 4": explicitrk4, "Explicit Midpoint": explicitmidpoint,
                             "Symplectic Forward Euler": sympforeuler, "Adaptive Heun-Euler": adaptiveheuneuler,
                             "Heun's": heuns, "Backward Euler": backeuler, "Euler-Trapezoidal": eulertrap,
                             "Predictor-Corrector Euler": eulertrap, "Implicit Midpoint": implicitmidpoint,
                             "Forward Euler": foreuler}
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

    def __init__(self, n=(1,), equ=(), y_i=(), t=(0, 0), savetraj=0, stpsz=1.0, eta=0, **consts):
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
        self.equ = list(equ)
        self.y = [numpy.resize(i, n) for i in y_i]
        self.dim = tuple([1] + list(n))
        self.t = t[0]
        self.t0 = t[0]
        self.t1 = t[1]
        self.soln = [[numpy.resize(i, n)] for i in self.y]
        # noinspection PyTypeChecker
        self.soln.append([t[0]])
        self.consts = consts
        for k in self.consts:
            self.consts.update({k: numpy.resize(self.consts[k], n)})
        self.traj = savetraj
        self.method = None
        if (stpsz < 0 < t[0] - t[1]) or (stpsz > 0 > t[0] - t[1]):
            self.dt = -1 * stpsz
        else:
            self.dt = stpsz
        self.eta = eta
        self.eqnum = len(equ)

    def chgendtime(self, t):
        self.t1 = t

    def chgbegtime(self, t):
        self.t0 = t

    def chgcurtime(self, t):
        self.t = t

    def chgtime(self, t=()):
        if len(t) == 1:
            warning("You have passed an array of that only contains one element, "
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
        self.dt = h

    @staticmethod
    def availmethods():
        print(available_methods.keys())
        return available_methods

    def setmethod(self, method):
        if method in available_methods.keys():
            self.method = method
        else:
            print("The method you selected does not exist in the list of available method, "
                  "call availmethods() to see what these are")

    def addequ(self, eq, ic):
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
        if len(indices) > self.eqnum:
            raise LengthError("You've specified the removal of more equations than there exists!")
        for i in indices:
            self.y.pop(i)
            self.soln.pop(i)
            self.equ.pop(i)
        self.eqnum -= len(indices)

    def showequ(self):
        for i in range(self.eqnum):
            print("dy_{} = ".format(i) + self.equ[i])
        return self.equ

    def numequ(self):
        print(self.eqnum)
        return self.eqnum

    def initcond(self):
        for i in range(self.eqnum):
            print("y_{}({}) = {}".format(i, self.t0, self.y[i]))
        return self.y

    def finalcond(self, p=1):
        if p:
            for i in range(self.eqnum):
                print("y_{}({}) = {}".format(i, self.t1, self.soln[i][-1]))
        return self.soln

    def showsys(self):
        for i in range(self.eqnum):
            print("Equation {}\ny_{}({}) = {}\ndy_{} = {}\ny_{}({}) = {}".format(i, i, self.t0, self.y[i], i,
                                                                                 self.equ[i], i, self.t,
                                                                                 self.soln[i][-1]))
        if self.consts:
            print("The constants that have been defined for this system are: ")
            print(self.consts)
        print("The time limits for this system are:\n "
              "t0 = {}, t1 = {}, t_current = {}, step_size = {}".format(self.t0, self.t1, self.t, self.dt))

    def addconsts(self, **additional_constants):
        self.consts.update({k: numpy.resize(additional_constants[k], self.dim) for k in additional_constants})

    def remconsts(self, **constants_removal):
        for i in constants_removal:
            if i in self.consts.keys():
                del self.consts[i]

    def chgdim(self, m=None):
        if m is not None:
            self.dim = tuple([1] + list(m))
            self.y = [numpy.resize(i, m) for i in self.y]
            solntime = self.soln[-1]
            self.soln = [[numpy.resize(i, m)] for i in self.soln[:-1]]
            self.soln.append(solntime)

    def reset(self, t=None):
        if t is not None:
            k = numpy.array(self.soln[-1])
            ind = numpy.argmin(numpy.square(numpy.subtract(k, t)))
            for i, k in enumerate(self.soln):
                self.soln[i] = list(numpy.delete(k, numpy.s_[ind + 1:], axis=0))
            self.t = t
        else:
            for i in range(self.eqnum + 1):
                if i < self.eqnum:
                    self.soln[i] = [self.y[i]]
                else:
                    self.soln[i] = 0
            self.t = 0

    def integrate(self, t=None):
        method = self.method
        if method is None:
            method = available_methods["Explicit Runge-Kutta 4"]
        else:
            try:
                method = available_methods[method]
            except KeyError:
                print("Method not available, defaulting to RK4 Method")
                method = available_methods["Explicit Runge-Kutta 4"]
        if not self.eta:
            eta = 0
        elif self.eta:
            import time as tm
            import sys
            eta = 1
        heff = [self.dt, self.dt]
        steps = 1
        if t:
            tf = t
        else:
            tf = self.t1
        steps_total = (tf - self.t) / heff[0]
        time_remaining = [0, 0]
        soln = self.soln
        vardict = {'t': self.t}
        vardict.update(self.consts)
        currenttime = self.t0
        while abs(vardict['t']) < abs(tf):
            try:
                if heff[0] != heff[1]:
                    steps_total *= heff[1] / heff[0]
                    heff[1] = heff[0]
                if eta == 1:
                    time_remaining[0] = tm.perf_counter()
                if "adaptive" not in method.__name__.lower().strip():
                    vardict.update(
                        {'t': currenttime})  # Avoids floating point precision errors by setting t to current time
                    currenttime = self.t0 + steps * heff[0]
                else:
                    currenttime = vardict['t']
                method(self.equ, vardict, soln, heff)
                if self.traj:
                    soln[-1].append(vardict['t'])
                else:
                    soln = [[i[-1]] for i in soln[:-1]]
                    soln.append([vardict['t']])
                if eta:
                    time_remaining[1] = 0.9 * time_remaining[1] + ((steps_total - steps) *
                                                                   0.1 * (tm.perf_counter() - time_remaining[0]))
                    sys.stdout.flush()
                    sys.stdout.write(
                        "\r{}% ----- ETA: {}".format(round(currenttime * 100 / (self.t1 - self.t0), ndigits=3),
                                                     "{} minutes and {} seconds".format(
                                                         int(time_remaining[1] / 60.),
                                                         int(time_remaining[1] - int(
                                                             time_remaining[1] / 60) * 60))))
                steps += 1
            except KeyboardInterrupt:
                break
        if eta:
            sys.stdout.flush()
            sys.stdout.write("\r100%  ")
        else:
            print("100%")
        self.soln = soln
        self.t = soln[-1][-1]
        self.dt = heff[0]
