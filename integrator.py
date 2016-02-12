import math
import numpy as np

safe_list = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 'degrees', 'e', 'exp', 'fabs', 'floor',
             'fmod', 'frexp', 'hypot', 'ldexp', 'log', 'log10', 'modf', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt',
             'tan', 'tanh']
safe_dict = {}

for k in safe_list:
    """
    Building a dictionary containing functions that are acceptable for use in the odes in order to make it more
    difficult for end-users to run malicious code.
    """
    safe_dict.update({'{}'.format(k): getattr(locals().get("math"), k)})
safe_dict.update({'abs': abs})


def bisectroot(equn, n, h, m, vardict, low, high, cstring, iterlimit=None):
    """
    Uses the bisection method to find the zeroes of the function defined in cstring.
    Designed to be used as a method to find the value of the next y_# in the implementation of the
    Backward Euler and Implicit Midpoint methods.
    """
    import copy as cpy
    if iterlimit is None:
        iterlimit = 52
    r = 0
    temp_vardict = cpy.deepcopy(vardict)
    temp_vardict.update({'t': vardict['t'] + m * h})
    while abs(high - low) > 10e-15 and r < iterlimit:
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
    aux = []
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        aux.append([0., 0., 0., 0.])
    for vari in range(eqnum):
        aux[vari][0] = seval(ode[vari], **vardict) * h[0]
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][0] * 0.5})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][1] = seval(ode[vari], **vardict) * h[0]
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][1] * 0.5})
    for vari in range(eqnum):
        aux[vari][2] = seval(ode[vari], **vardict) * h[0]
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][2]})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][3] = seval(ode[vari], **vardict) * h[0]
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][3]})
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1] +
                                             (aux[vari][0] + aux[vari][1] * 2 + aux[vari][2] * 2 + aux[vari][3]) / 6})
        soln[vari].append(vardict['y_{}'.format(vari)])


def explicitmidpoint(ode, vardict, soln, h):
    """
    Implementation of the Explicit Midpoint method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    aux = []
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        aux.append([0.])
    for vari in range(eqnum):
        aux[vari][0] = seval(ode[vari], **vardict) * h[0] + soln[vari][-1]
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][0] = seval(ode[vari], **vardict)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * aux[vari][0]})
        soln[vari].append(vardict['y_{}'.format(vari)])
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})


def implicitmidpoint(ode, vardict, soln, h):
    """
    Implementation of the Implicit Midpoint method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
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
        soln[vari].append(vardict['y_{}'.format(vari)])
    vardict.update({'t': vardict['t'] + h[0]})


def heuns(ode, vardict, soln, h):
    """
    Implementation of Heun's method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    aux = []
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        aux.append([0., 0.])
    for vari in range(eqnum):
        aux[vari][0] = seval(ode[vari], **vardict)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0] * h[0] + soln[vari][-1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][1] = seval(ode[vari], **vardict)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * (aux[vari][0] + aux[vari][1]) * 0.5})
        soln[vari].append(vardict['y_{}'.format(vari)])


def backeuler(ode, vardict, soln, h):
    """
    Implementation of the Implicit/Backward Euler method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
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
        soln[vari].append(vardict['y_{}'.format(vari)])
    vardict.update({'t': vardict['t'] + h[0]})


def foreuler(ode, vardict, soln, h):
    """
    Implementation of the Explicit/Forward Euler method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    aux = []
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        aux.append([0.])
    for vari in range(eqnum):
        aux[vari][0] = seval(ode[vari], **vardict) * h[0] + soln[vari][-1]
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
        soln[vari].append(vardict['y_{}'.format(vari)])
    vardict.update({'t': vardict['t'] + h[0]})


def eulertrap(ode, vardict, soln, h):
    """
    Implementation of the Euler-Trapezoidal method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    aux = []
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        aux.append([0., 0., 0.])
    for vari in range(eqnum):
        aux[vari][0] = seval(ode[vari], **vardict)
    for vari in range(eqnum):
        aux[vari][1] = seval(ode[vari], **vardict) * h[0] + soln[vari][-1]
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): aux[vari][1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][2] = seval(ode[vari], **vardict)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * (aux[vari][0] + aux[vari][2])})
        soln[vari].append(vardict['y_{}'.format(vari)])


def adaptiveheuneuler(ode, vardict, soln, h, tol=0):
    """
    Implementation of the Adaptive Heun-Euler method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    if tol == 0:
        tol = 10e-6
    aux = []
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        aux.append([0., 0.])
    for vari in range(eqnum):
        aux[vari][0] = seval(ode[vari], **vardict)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0] * h[0] + soln[vari][-1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][1] = seval(ode[vari], **vardict)
    err = 0.
    for vari in range(eqnum):
        temp_error = np.absolute(np.subtract(aux[vari][0] * h[0], (aux[vari][0] + aux[vari][1]) * 0.5 * h[0]))
        if temp_error > err:
            err = temp_error
    if err > tol:
        h[0] /= 2
        adaptiveheuneuler(ode, vardict, soln, h)
    else:
        if 64 * err < tol:
            h[0] *= 2
        for vari in range(eqnum):
            vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * aux[vari][0]})
            soln[vari].append(vardict['y_{}'.format(vari)])


def sympforeuler(ode, vardict, soln, h):
    """
    Implementation of the Symplectic Euler method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    aux = []
    eqnum = len(ode)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        aux.append([0.])
    for vari in range(eqnum):
        aux[vari][0] = seval(ode[vari], **vardict) * h[0] + soln[vari][-1]
        if vari % 2 == 0:
            vardict.update({"y_{}".format(vari): aux[vari][0]})
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
        soln[vari].append(vardict['y_{}'.format(vari)])
    vardict.update({'t': vardict['t'] + h[0]})


available_methods = {"Explicit Runge-Kutta 4": explicitrk4, "Explicit Midpoint": explicitmidpoint,
                     "Symplectic Forward Euler": sympforeuler, "Adaptive Heunn-Euler": adaptiveheuneuler,
                     "Heun's": heuns, "Backward Euler": backeuler, "Euler-Trapezoidal": eulertrap,
                     "Predictor-Corrector Euler": eulertrap, "Implicit Midpoint": implicitmidpoint,
                     "Forward Euler": foreuler}


def mpintegrator(ode, yi, t, h, method=None, eta=None):
    if method is None:
        method = available_methods["Explicit Runge-Kutta 4"]
    else:
        try:
            method = available_methods[method]
        except KeyError:
            print("Method not available, defaulting to RK4 Method")
            method = available_methods["Explicit Runge-Kutta 4"]
    if eta is None:
        eta = 0
    elif eta == 1:
        import time as tm
        import sys
    heff = [math.copysign(h, t[1] - t[0])]
    steps = 1
    steps_total = (t[1] - t[0]) / heff[0]
    time_remaining = [0, 0]
    soln = yi
    soln.append([t[0]])
    vardict = {'t': t[0]}
    currenttime = t[0]
    while abs(vardict['t']) < abs(t[1]):
        if eta == 1:
            time_remaining[0] = tm.perf_counter()
        if method != adaptiveheuneuler:
            vardict.update({'t': currenttime})  # Avoids floating point precision errors by setting t to current time
            currenttime = t[0] + steps * heff[0]
        else:
            currenttime = vardict['t']
        method(ode, vardict, soln, heff)
        soln[-1].append(vardict['t'])
        if eta:
            time_remaining[1] = 0.9 * time_remaining[1] + ((steps_total - steps) *
                                                           0.1 * (tm.perf_counter() - time_remaining[0]))
            sys.stdout.flush()
            sys.stdout.write("\r{}% ----- ETA: {}".format(round(currenttime * 100 / (t[1] - t[0]), ndigits=3),
                                                          "{} minutes and {} seconds".format(
                                                              int(time_remaining[1] / 60.),
                                                              int(time_remaining[1] - int(
                                                                  time_remaining[1] / 60) * 60))))
        steps += 1
    if eta:
        sys.stdout.flush()
        sys.stdout.write("\r100%  ")
    else:
        print("100%")
    return soln


def main():
    import os

    n = int(input("Please enter the order of the system: N = "))
    if input("Would you like to enter the system in vector form? ").replace(" ", "").lower() in ["yes", "1", "y"]:
        vectorised = 1
    else:
        vectorised = 0
    eqn = []
    y_i = []
    stpsz = 1
    tlim = [0., 1.]
    if vectorised == 0:
        for i in range(n):
            eqn.append(input("Please enter the ode using y_# and t as the variables: y_" + str(i) + "' = "))
            eqn[i] = eqn[i].replace('^', '**')
            print(eqn)
            y_i.append([float(input("Please enter the initial value for this variable: y(" + str(i) + ") = "))])
            tlim = [float(input("Please enter the initial time: t_initial = ")),
                    float(input("Please enter the final time: t_final = "))]
            stpsz = float(input("Please enter a step size for the integration: h = "))
    else:
        eqn = [equn.replace("^", "**") for equn in input("Please enter each ode separated by a comma using y_n as the "
                                                         "variables for n between 0 and {}: "
                                                         "\n".format(n - 1)).replace(" ", "").split(",")]
        print(eqn)
        while len(eqn) != n:
            eqn = [equn.replace("^", "**") for equn in input("You have entered {} odes when {} odes were required. "
                                                             "\n".format(len(eqn), n)).replace(" ", "").split(",")]
        y_i = [[float(y_initial)] for y_initial in input("Please enter the initial values separated by a comma for each"
                                                         " y_n for n between "
                                                         "0 and {}: \n".format(n - 1)).replace(" ", "").split(",")]
        while len(y_i) != n:
            y_i = [[float(y_initial)] for y_initial in input("You have entered {} initial conditions "
                                                             "when {} initial conditions were required. "
                                                             "\n".format(len(y_i), n)).replace(" ", "").split(",")]
        time_param = input("Please enter the lower and upper time limits, and step size separated by commas: "
                           "\n").replace(" ", "").split(",")
        while len(time_param) != 3:
            time_param = input("You have entered {} parameters when 3 were required. "
                               "\n".format(len(time_param))).replace(" ", "").split(",")
        tlim = [float(time_param[0]), float(time_param[1])]
        stpsz = float(time_param[2])
    if not input("Would you like to explicitly choose the method of integration? ").replace(' ', '').lower() in \
            ["no", "0", "do not do this to me", ""]:
        print("Choose from one of the following methods: ")
        shorthand = {"RK4": "Explicit Runge-Kutta 4", "Euler": "Forward Euler"}
        c = 1
        for keys in sorted(available_methods.keys()):
            print("{}: {}".format(c, keys), end=', ')
            shorthand.update({str(c): keys})
            c += 1
        print("\n")
        intmethod = input("Please enter the name of the method you would like to use, "
                          "you may use numbers to refer to the methods available: ")
        if intmethod in shorthand:
            intmethod = shorthand[intmethod]
        while intmethod not in available_methods.keys():
            intmethod = input("That is not an available method, "
                              "please enter the name of the method you would like to use: ")
            if intmethod in shorthand:
                intmethod = shorthand[intmethod]
    else:
        intmethod = "Explicit Runge-Kutta 4"
    plotbool = input("Would you like to make some basic plots? (Yes: 1, No: 0) : ")
    savedir = input("Please enter the path to save data and plots to: ")

    reservedchars = ['<', '>', '"', '/', '|', '?', '*']

    for i in reservedchars:
        while i in savedir:
            print("Your save directory contains one of the following invalid characters{}".format(reservedchars), )
            savedir = input(" please enter a new save directory: ")
    if savedir[-1] != "\ ".strip():
        savedir += "\ ".strip()

    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    res = mpintegrator(eqn, y_i, tlim, stpsz, method=intmethod, eta=0)

    vardicto = {}
    vardicto.update({'t': res[-1]})

    for i in range(n):
        vardicto.update({'y_{}'.format(i): res[i]})

    for i in range(n):
        file = open("{}{}.txt".format(savedir, "y_{}".format(i)), 'w')
        file.truncate()
        file.write((str(vardicto["y_{}".format(i)]).replace("[", "")).replace("]", ""))
        file.close()

    file = open("{}time.txt".format(savedir), 'w')
    file.truncate()
    file.write(str(vardicto["t"]))
    file.close()

    initial_cond = ""

    for kp in range(n):
        initial_cond += "y_{}({}) = {}\n".format(kp, tlim[0], y_i[kp][0])
    for kp in range(2):
        initial_cond += "t_{} = {}\n".format(kp, tlim[kp])
    initial_cond += "Step Size - h = {}\n".format(stpsz)

    file = open("{}initialcond.txt".format(savedir), 'w')
    file.truncate()
    file.write(initial_cond)
    file.close()

    if plotbool.strip() == "1":
        import matplotlib.pyplot as plt
        import matplotlib as mplt

        font = {'family': 'serif',
                'weight': 'normal',
                'size': 6}
        mplt.rc('font', **font)
        temp = input("Please enter variables as pairs (x-axis first) separated by commas eq. y_0, y_1; t, y_0: ")
        temp = temp.replace(" ", "")
        titles = []
        plotvars = []
        pltcount = len(temp.split(';'))
        scatter = []
        mu = []

        if temp != "":
            for i in range(pltcount):
                print("For plot {} please enter the following: ".format(i))
                titles.append([input("Plot Title: "), input("x-axis Title: "), input("y-axis Title: ")])
                scatter.append(input("Scatter(S) or Line(L) plot: "))
                p = float(input("Please enter the period of sampling where the period is between {} and {}: P ="
                                " ".format(math.copysign(stpsz, tlim[1] - tlim[0]), tlim[1] - tlim[0])))
                mu.append(int(abs(p / stpsz)))
                plotvars.append([vardicto[temp.split(';')[i].split(',')[0]],
                                 vardicto[temp.split(';')[i].split(',')[1]]])

            fig = []

            for i in range(pltcount):
                fig.append(plt.figure(i, figsize=(24, 24), dpi=600))
                ax = fig[i].add_subplot(111)
                if scatter[i].strip().lower() == "s":
                    ax.scatter(plotvars[i][0][0:-1:mu[i]], plotvars[i][1][0:-1:mu[i]], s=0.25, marker='.')
                else:
                    ax.plot(plotvars[i][0][0:-1:mu[i]], plotvars[i][1][0:-1:mu[i]])
                ax.set_title(titles[i][0])
                ax.set_xlabel(titles[i][1])
                ax.set_ylabel(titles[i][2])
                ax.grid(True, which='both', linewidth=0.075)
                fig[i].savefig("{}{}.pdf".format(savedir, titles[i][0]), bbox_inches='tight')
                fig[i].savefig("{}{}.png".format(savedir, titles[i][0]), bbox_inches='tight')


if __name__ == "__main__":
    main()
