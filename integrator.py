import odesolver


def main():
    import os
    import math
    import sys

    if len(sys.argv) == 1:
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
                y_i.append(float(input("Please enter the initial value for this variable: y(" + str(i) + ") = ")))
                tlim = [float(input("Please enter the initial time: t_initial = ")),
                        float(input("Please enter the final time: t_final = "))]
                stpsz = float(input("Please enter a step size for the integration: h = "))
        else:
            eqn = [equn.replace("^", "**") for equn in input("Please enter each ode separated by a comma using y_n as "
                                                             "the variables for n between 0 and {}: "
                                                             "\n".format(n - 1)).replace(" ", "").split(",")]
            print(eqn)
            while len(eqn) != n:
                eqn = [equn.replace("^", "**") for equn in input("You have entered {} odes when {} odes were required. "
                                                                 "\n".format(len(eqn), n)).replace(" ", "").split(",")]
            y_i = [float(y_initial) for y_initial in input("Please enter the initial values separated by a comma for "
                                                           "each y_n for n between "
                                                           "0 and {}: \n".format(n - 1)).replace(" ", "").split(",")]
            while len(y_i) != n:
                y_i = [float(y_initial) for y_initial in input("You have entered {} initial conditions "
                                                               "when {} initial conditions were required. "
                                                               "\n".format(len(y_i), n)).replace(" ", "").split(",")]
            time_param = input("Please enter the lower and upper time limits, and step size separated by commas: "
                               "\n").replace(" ", "").split(",")
            while len(time_param) != 3:
                time_param = input("You have entered {} parameters when 3 were required. "
                                   "\n".format(len(time_param))).replace(" ", "").split(",")
            tlim = [float(time_param[0]), float(time_param[1])]
            stpsz = float(time_param[2])
    else:
        argu = sys.argv
        try:
            y_i = [float(i) for i in argu[argu.index('-y_i') + 1:argu.index('-tp')]]
            eqn = [i for i in argu[argu.index('-eqn') + 1:argu.index('-y_i')]]
            tlim = [float(i) for i in argu[argu.index('-tp') + 1:argu.index('-o')]]
            n = len(eqn)
        except ValueError:
            raise ValueError

    intobj = odesolver.DifferentialSystem(eqn, y_i, tlim, savetraj=1, stpsz=stpsz, eta=1)
    if '-m' not in sys.argv:
        if not input("Would you like to explicitly choose the method of integration? ").replace(' ', '').lower() in \
                ["no", "0", "do not do this to me", ""]:
            print("Choose from one of the following methods: ")
            shorthand = {"RK4": "Explicit Runge-Kutta 4", "Euler": "Forward Euler"}
            c = 1
            for keys in sorted(odesolver.available_methods.keys()):
                print("{}: {}".format(c, keys), end=', ')
                shorthand.update({str(c): keys})
                c += 1
            print("\n")
            intmethod = input("Please enter the name of the method you would like to use, "
                              "you may use numbers to refer to the methods available: ")
            if intmethod in shorthand:
                intmethod = shorthand[intmethod]
            while intmethod not in odesolver.available_methods.keys():
                intmethod = input("That is not an available method, "
                                  "please enter the name of the method you would like to use: ")
                if intmethod in shorthand:
                    intmethod = shorthand[intmethod]
        else:
            intmethod = "Explicit Runge-Kutta 4"
    else:
        intmethod = sys.argv[sys.argv.index('-m') + 1]
    plotbool = input("Would you like to make some basic plots? (Yes: 1, No: 0) : ")
    if len(sys.argv) > 1:
        savedir = sys.argv[sys.argv.index('-o') + 1]
    else:
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

    intobj.integrate(method=intmethod)

    res = intobj.finalcond(p=0)
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
        initial_cond += "y_{}({}) = {}\n".format(kp, tlim[0], y_i[kp])
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
