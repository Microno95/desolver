import math
from desolver import backend as D
from desolver.integrators.integrator_types import TableauIntegrator

def implicit_aware_update_timestep(integrator: TableauIntegrator):
    timestep_from_error, redo_step = integrator.update_timestep(ignore_custom_adaptation=True)
    if "niter0" in integrator.solver_dict.keys():
        # Adjust the timestep according to the computational cost of 
        # solving the nonlinear system at each timestep
        if integrator.solver_dict['niter0'] != 0 and integrator.solver_dict['niter1'] != 0:
            Tk0, CTk0 = D.ar_numpy.log(integrator.solver_dict['tau0']), math.log(integrator.solver_dict['niter0'])
            Tk1, CTk1 = D.ar_numpy.log(integrator.solver_dict['tau1']), math.log(integrator.solver_dict['niter1'])
            dnCTk = CTk1 - CTk0
            ddCTk = Tk1 - Tk0
            if ddCTk > 0:
                dCTk = dnCTk / ddCTk
            else:
                dCTk = D.ar_numpy.zeros_like(integrator.solver_dict['timestep'])
            tau2 = D.ar_numpy.exp(-dCTk)
        else:
            tau2 = None
        # ---- #
        # Adjust the timestep according to the precision achieved by the 
        # nonlinear system solver at each timestep
        total_error_tolerance = integrator.solver_dict['atol'] + integrator.solver_dict['rtol']
        tau3 = D.ar_numpy.ones_like(integrator.solver_dict['timestep'])
        if integrator.solver_dict['newton_prec1'] > 0.0:
            with D.numpy.errstate(divide='ignore'):
                epsilon_current = total_error_tolerance / integrator.solver_dict['newton_prec1']
            tau3 = tau3*D.ar_numpy.where(D.ar_numpy.isfinite(epsilon_current), epsilon_current, 1.0)
        if integrator.solver_dict['newton_prec0'] > 0.0:
            with D.numpy.errstate(divide='ignore'):
                epsilon_last = total_error_tolerance / integrator.solver_dict['newton_prec0']
            tau3 = tau3*D.ar_numpy.where(D.ar_numpy.isfinite(epsilon_last), epsilon_last, 1.0)
        # ---- #
        if tau2 is None:
            tau = tau3
        else:
            tau = D.ar_numpy.minimum(tau2, tau3)
        tau = (1 + 0.1*D.ar_numpy.arctan((tau - 1)/0.1))
        return tau * timestep_from_error, redo_step
    else:
        return timestep_from_error, redo_step
