import numpy as np

error_coeff_arrayrk45ck = [[-0.0042937748015873],
                         [ 0.                ],
                         [ 0.0186685860938579],
                         [-0.0341550268308081],
                         [-0.0193219866071429],
                         [ 0.0391022021456804]]

# Based on arXiv:1501.04345v2 - BAB's9o7H
BABPrimes9o7H_coefficients = [[0.04649290043965892,
                               0.154901012702888,
                               0.31970548287359174,
                               -0.19292000881571322,
                               0.17182061279957458,
                               0.17182061279957458,
                               -0.19292000881571322,
                               0.31970548287359174,
                               0.154901012702888,
                               0.04649290043965892],
                              [0.1289555065927298,
                               0.10907642985488271,
                               -0.013886035680471514,
                               0.18375497456418036,
                               0.18419824933735726,
                               0.18375497456418036,
                               -0.013886035680471514,
                               0.10907642985488271,
                               0.1289555065927298,
                               0.0000000000000000]]

# Based on arXiv:1501.04345v2 - ABAs5o6H
ABAs5o6HA_coefficients = [[0.15585935917621682,
                          -0.007025499091957318,
                          0.35116613991574047,
                          0.35116613991574047,
                          -0.007025499091957318,
                          0.15585935917621682],
                         [-0.6859195549562167,
                          0.9966295909529364,
                          0.3785799280065607,
                          0.9966295909529364,
                          -0.6859195549562167,
                          0.0]]

def explicitrk4(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Explicit Runge-Kutta 4 method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """

    dim = [eqnum, 4]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][0] * 0.5})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][1] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][1] * 0.5})
    for vari in range(eqnum):
        aux[vari][2] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][2]})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][3] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][3]})
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): (soln[vari][-1] +
                                              (aux[vari][0] + aux[vari][1] * 2 + aux[vari][2] * 2 + aux[vari][3]) / 6)})
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))


def explicitgills(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Explicit Runge-Kutta 4 method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    dim = [eqnum, 4]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][0] * 0.5})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][1] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][0] * 0.4142135623730950 +
                                             aux[vari][1] * 0.2928932188134524})
    for vari in range(eqnum):
        aux[vari][2] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][2]})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][3] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] - aux[vari][1] * 0.7071067811865475 +
                                             aux[vari][2] * 1.7071067811865475})
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): (soln[vari][-1] +
                                              (aux[vari][0] + aux[vari][1] * 0.585786437626905 +
                                               aux[vari][2] * 3.4142135623730950 + aux[vari][3]) / 6)})
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))


def explicitrk45ck(ode, vardict, soln, h, relerr, eqnum, tol=0.5):
    """
    Implementation of the Explicit Runge-Kutta-Fehlberg method.
    Ode is a list of strings with the expressions defining the odes.
    Vardict is a dictionary containing the current variables.
    Soln is the list containing the computed values for the odes.
    h is the step-size in computing the next value of the variable(s)
    """
    dim = [eqnum, 6]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    t_initial = vardict['t']
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + aux[vari][0] / 5})
    vardict.update({'t': t_initial + h[0] / 5})
    for vari in range(eqnum):
        aux[vari][1] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + 3.0 * aux[vari][0] / 40 + 9.0 * aux[vari][1] / 40})
    vardict.update({'t': t_initial + 3 * h[0] / 10})
    for vari in range(eqnum):
        aux[vari][2] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + (3.0 * aux[vari][0] - 9.0 * aux[vari][1] +
                                                               12.0 * aux[vari][2]) / 10})
    vardict.update({'t': t_initial + 3 * h[0] / 5})
    for vari in range(eqnum):
        aux[vari][3] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): (soln[vari][-1] - 11.0 * aux[vari][0] / 54 - 5.0 * aux[vari][1] / 2 -
                                              70.0 * aux[vari][2] / 27 + 35.0 * aux[vari][3] / 27)})
    vardict.update({'t': t_initial + h[0]})
    for vari in range(eqnum):
        aux[vari][4] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): (soln[vari][-1] +
                                              1631.0 * aux[vari][0] / 55296 + 175.0 * aux[vari][1] / 512 +
                                              575.0 * aux[vari][2] / 13824 + 44275.0 * aux[vari][3] / 110592 +
                                              253.0 * aux[vari][4] / 4096)})
    vardict.update({'t': t_initial + 7.0 * h[0] / 8})
    for vari in range(eqnum):
        aux[vari][5] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    coeff = []
    for vari in range(eqnum):
        coeff.append([])
        coeff[vari].append(soln[vari][-1] + 37.0 * aux[vari][0] / 378 + 250.0 * aux[vari][2] / 621 +
                           125.0 * aux[vari][3] / 594 + 512.0 * aux[vari][5] / 1771)
        coeff[vari].append(soln[vari][-1] + 2825.0 * aux[vari][0] / 27648 + 18575.0 * aux[vari][2] / 48384 +
                           13525.0 * aux[vari][3] / 55296 + 277.0 * aux[vari][4] / 14336 + aux[vari][5] / 4.0)
    error_coeff_array = [np.resize(i, dim) for i in error_coeff_arrayrk45ck]
    err_estimate = np.abs(
        np.ravel([np.sum(aux[vari] * error_coeff_array, axis=0) for vari in range(eqnum)])).max()
    vardict.update({'t': t_initial + h[0]})
    if err_estimate != 0:
        h[1] = h[0]
        corr = h[0] * tol * (relerr * h[0] / err_estimate) ** (1.0 / 4.0)
        if abs(corr + t_initial) < abs(h[2]):
            if corr != 0:
                h[0] = corr
        else:
            h[0] = abs(h[2] - t_initial)
    if err_estimate > relerr * h[0] / (tol ** 3):
        for vari in range(eqnum):
            vardict.update({'y_{}'.format(vari): soln[vari][-1]})
        vardict.update({'t': t_initial})
        explicitrk45ck(ode, vardict, soln, h, relerr, eqnum)
    else:
        for vari in range(eqnum):
            vardict.update({'y_{}'.format(vari): coeff[vari][1]})
            pt = soln[vari]
            kt = np.array([coeff[vari][1]])
            soln[vari] = np.concatenate((pt, kt))


def explicitmidpoint(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Explicit Midpoint method.
    """

    dim = [eqnum, 2]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize([seval(ode[vari], **vardict) * h[0] + soln[vari][-1]], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize([seval(ode[vari], **vardict)], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): np.array(soln[vari][-1] + h[0] * aux[vari][0])})
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + 0.5 * h[0]})


def implicitmidpoint(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Implicit Midpoint method.
    """

    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        bisectroot(ode[vari], vari, h, 0.5, vardict, - soln[vari][-1] - h[0] * seval(ode[vari], **vardict),
                   soln[vari][-1] + h[0] * seval(ode[vari], **vardict),
                   cstring="temp_vardict['y_{}'.format(n)] - vardict['y_{}'.format(n)] - "
                           "h[0] * 0.5 * seval(equn, **vardict)")
    for vari in range(eqnum):
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def heuns(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of Heun's method.
    """

    dim = [eqnum, 2]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0] * h[0] + soln[vari][-1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][1] = np.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * (aux[vari][0] + aux[vari][1]) * 0.5})
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))


def backeuler(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Implicit/Backward Euler method.
    """

    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        bisectroot(ode[vari], vari, h, 1.0, vardict, - soln[vari][-1] - h[0] * seval(ode[vari], **vardict),
                   soln[vari][-1] + h[0] * seval(ode[vari], **vardict),
                   cstring="temp_vardict['y_{}'.format(n)] - h[0] * seval(equn, **temp_vardict) "
                           "- vardict['y_{}'.format(n)]")
    for vari in range(eqnum):
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def foreuler(ode, variable_states, h, relerr):
    """
    Implementation of the Explicit/Forward Euler method.
    """



    dim = [eqnum, 1]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize(seval(ode[vari], **vardict) * h[0] + soln[vari][-1], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def impforeuler(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of an Improved Forward Euler method.
    """

    dim = [eqnum, 2]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0] + soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][1] = np.resize(seval(ode[vari], **vardict) * h[0], dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + 0.5 * (aux[vari][0] + aux[vari][1])})
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def eulertrap(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Euler-Trapezoidal method.
    """

    dim = [eqnum, 3]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        aux[vari][1] = np.resize(seval(ode[vari], **vardict) * h[0] + soln[vari][-1], dim)
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): aux[vari][1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][2] = np.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): soln[vari][-1] + h[0] * (aux[vari][0] + aux[vari][2])})
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))


def adaptiveheuneuler(ode, vardict, soln, h, relerr, eqnum, tol=0.9):
    """
    Implementation of the Adaptive Heun-Euler method.
    """

    dim = [eqnum, 2]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)

    dim = soln[0][0].shape

    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        aux[vari][0] = np.resize(seval(ode[vari], **vardict), dim)
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0] * h[0] + soln[vari][-1]})
    vardict.update({'t': vardict['t'] + h[0]})
    for vari in range(eqnum):
        aux[vari][1] = np.resize(seval(ode[vari], **vardict), dim)
    err = [(aux[vari][0] - aux[vari][1]) * h[0] for vari in range(eqnum)]
    err = np.abs(err).max()
    err *= ((h[2] - vardict['t'] + h[0]) / h[0])
    if err >= relerr:
        vardict.update({'t': vardict['t'] - h[0]})
        if err != 0:
            h[0] *= tol * (relerr / err)
        adaptiveheuneuler(ode, vardict, soln, h, relerr, eqnum)
    else:
        if err < relerr and err != 0:
            h[0] *= (relerr / err) ** (1.0 / 2.0)
        for vari in range(eqnum):
            vardict.update({"y_{}".format(vari): soln[vari][-1] + (aux[vari][0] + aux[vari][1]) * 0.5 * h[0]})
            pt = soln[vari]
            kt = np.array([vardict['y_{}'.format(vari)]])
            soln[vari] = np.concatenate((pt, kt))


def sympforeuler(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Symplectic Euler method.
    """

    dim = [eqnum, 1]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for vari in range(eqnum):
        if vari % 2 == 0:
            aux[vari][0] = np.resize((seval(ode[vari], **vardict) * h[0] * 0.5 + soln[vari][-1]), dim)
            vardict.update({"y_{}".format(vari): aux[vari][0]})
    for vari in range(eqnum):
        if vari % 2 == 1:
            aux[vari][0] = np.resize((seval(ode[vari], **vardict) * h[0] + soln[vari][-1]), dim)
            vardict.update({"y_{}".format(vari): aux[vari][0]})
    for vari in range(eqnum):
        if vari % 2 == 0:
            aux[vari][0] = np.resize((seval(ode[vari], **vardict) * h[0] * 0.5 + soln[vari][-1]), dim)
            vardict.update({"y_{}".format(vari): aux[vari][0]})
    for vari in range(eqnum):
        vardict.update({"y_{}".format(vari): aux[vari][0]})
        pt = soln[vari]
        kt = np.array([vardict['y_{}'.format(vari)]])
        soln[vari] = np.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def sympBABs9o7H(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Symplectic BAB's9o7H method based on arXiv:1501.04345v2
    """

    dim = [eqnum, 1]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for stage in range(0, len(BABPrimes9o7H_coefficients[0])):
        for vari in range(1, eqnum, 2):
            aux[vari][0] = np.resize((vardict["y_{}".format(vari)] +
                                         BABPrimes9o7H_coefficients[0][stage] * seval(ode[vari], **vardict) * h[0] / 2),
                                        dim)
        for vari in range(1, eqnum, 2):
            vardict.update({"y_{}".format(vari): aux[vari][0]})
        for vari in range(0, eqnum, 2):
            aux[vari][0] = np.resize((vardict["y_{}".format(vari)] +
                                         BABPrimes9o7H_coefficients[1][stage] * seval(ode[vari], **vardict) * h[0]),
                                        dim)
        for vari in range(0, eqnum, 2):
            vardict.update({"y_{}".format(vari): aux[vari][0]})
        for vari in range(1, eqnum, 2):
            aux[vari][0] = np.resize((vardict["y_{}".format(vari)] +
                                         BABPrimes9o7H_coefficients[0][stage] * seval(ode[vari], **vardict) * h[0] / 2),
                                        dim)
        for vari in range(1, eqnum, 2):
            vardict.update({"y_{}".format(vari): aux[vari][0]})
    for vari in range(eqnum):
        pt = soln[vari]
        kt = np.array([aux[vari][0]])
        soln[vari] = np.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})


def sympABAs5o6HA(ode, vardict, soln, h, relerr, eqnum):
    """
    Implementation of the Symplectic ABAs5o6HA method based on arXiv:1501.04345v2
    """

    dim = [eqnum, 1]
    dim.extend(soln[0][0].shape)
    dim = tuple(dim)
    if np.iscomplexobj(soln[0]):
        aux = np.resize([0. + 0j], dim)
    else:
        aux = np.resize([0.], dim)
    dim = soln[0][0].shape
    for vari in range(eqnum):
        vardict.update({'y_{}'.format(vari): soln[vari][-1]})
    for stage in range(0, len(ABAs5o6HA_coefficients[0])):
        for vari in range(0, eqnum, 2):
            aux[vari][0] = np.resize((vardict["y_{}".format(vari)] +
                                         ABAs5o6HA_coefficients[1][stage] * seval(ode[vari], **vardict) * h[0] / 2),
                                        dim)
        for vari in range(0, eqnum, 2):
            vardict.update({"y_{}".format(vari): aux[vari][0]})
        for vari in range(1, eqnum, 2):
            aux[vari][0] = np.resize((vardict["y_{}".format(vari)] +
                                         ABAs5o6HA_coefficients[0][stage] * seval(ode[vari], **vardict) * h[0]), dim)
        for vari in range(1, eqnum, 2):
            vardict.update({"y_{}".format(vari): aux[vari][0]})
        for vari in range(0, eqnum, 2):
            aux[vari][0] = np.resize((vardict["y_{}".format(vari)] +
                                         ABAs5o6HA_coefficients[1][stage] * seval(ode[vari], **vardict) * h[0] / 2),
                                        dim)
        for vari in range(0, eqnum, 2):
            vardict.update({"y_{}".format(vari): aux[vari][0]})
    for vari in range(eqnum):
        pt = soln[vari]
        kt = np.array([aux[vari][0]])
        soln[vari] = np.concatenate((pt, kt))
    vardict.update({'t': vardict['t'] + h[0]})