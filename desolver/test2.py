import types
import desolver as de
import matplotlib.pyplot as plt
import timeit
import numpy
import argparse
plt.ion()
de.init_module(True)

parser = argparse.ArgumentParser(description="Numerically integrate the trajectory of a nonlinear 1D oscillator using a variety of integration methods and parameters.")
parser.add_argument("-dt", "--stepsize", type=float, default=0.0625/8, help="Set the step size of the integration")
parser.add_argument("-m", "--mass", type=float, default=1.0, help="Set the mass of the oscillator")
parser.add_argument("-k", "--stiffness", type=float, default=1.0, help="Set the spring stiffness")
parser.add_argument("-b", "--lineardamping", type=float, default=0.0, help="Set the linear damping coefficient")
parser.add_argument("-x0", "--initialdisp", type=float, default=1.0, help="Set the initial displacement")
parser.add_argument("-vx0", "--initialspeed", type=float, default=0.0, help="Set the initial speed")
parser.add_argument("-nl", "--nonlinearamp", type=float, default=0.0, help="Set the amplitude of the nonlinear driving forces")
parser.add_argument("-nlw", "--nonlinearfreq", type=float, default=100.0, help="Set the frequency of the nonlinear driving forces")
parser.add_argument("-ti", "--initialtime", type=float, default=0.0, help="Set the initial integration time")
parser.add_argument("-tf", "--finaltime", type=float, default=1e3, help="Set the final integration time")
parser.add_argument("-im", "--integrationmethod", type=str, default="ABAS5O6HA", help="Set the final integration time")
parser.add_argument("--show_methods", action="store_true", help="Show the available integration methods")
args = parser.parse_args()

if args.show_methods:
    print("The methods of integration available are the following:")
    print(list(de.OdeSystem.available_methods(True).keys()))
    exit()

def test_func(plot=False):
    a = de.OdeSystem(n=(1,), equ=(("y_1/m", args.initialdisp), ("-k*y_0 - 2*b*y_1 + k*X_0*cos(w*t) + k*X_0*0.5*sin(w*t*0.33)", args.initialspeed)), savetraj=True, constants={'m': args.mass, 'k': args.stiffness, 'b': args.lineardamping, 'X_0': args.nonlinearamp, 'w': args.nonlinearfreq}, stpsz=args.stepsize, t=[args.initialtime, args.finaltime], relerr=1e-1)

    a.available_methods()
    a.set_method("ABAS5O6HA")
    print(a.get_method())

    if plot:
        global fig, ax
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax2.axhline(1.0)
        fig.canvas.draw()
        background1 = fig.canvas.copy_from_bbox(ax.bbox)
        background2 = fig.canvas.copy_from_bbox(ax2.bbox)

        scatter1_, = ax.plot([0], [0], '-r',linewidth=0.75)
        scatter2_, = ax2.plot([0], [0], '--b',linewidth=0.75)
        fig.canvas.draw()

        plt.show()

        def cb(de_sys):
            if len(a.get_sample_times()) % 100 == 0 or abs(a.get_current_time() - a.get_end_time()) < a.get_step_size():
                global ax, fig
                fig.canvas.restore_region(background1)
                fig.canvas.restore_region(background2)
                scatter1_.set_data(de_sys.soln[0], de_sys.soln[1])
                scatter2_.set_data(de_sys.sample_times, ((de_sys.consts['k'] * de_sys.soln[0] ** 2) / 2 + (
                de_sys.consts['m'] * de_sys.soln[1] ** 2) / 2)/((de_sys.consts['k'] * a.y[0] ** 2) / 2 + (de_sys.consts['m'] * a.y[1] ** 2) / 2))
                ax.set_title("{}".format(de_sys.t))
                ax.draw_artist(scatter1_)
                ax2.draw_artist(scatter2_)
                ax.relim()
                ax2.relim()
                ax.autoscale_view()
                ax2.autoscale_view()
                fig.canvas.blit(ax.bbox)
                fig.canvas.blit(ax2.bbox)
                plt.pause(1e-10)
    else:
        def cb(de_sys):
            return

    a.integrate(callback=cb)
    if plot: plt.show(block=True)

test_func(plot=True)
