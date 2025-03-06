import math
from desolver import backend as D
from desolver.integrators.integrator_types import TableauIntegrator

def implicit_aware_update_timestep(integrator: TableauIntegrator):
    timestep = integrator.solver_dict['timestep']
    safety_factor = integrator.solver_dict['safety_factor']
    timestep_from_error, redo_step = integrator.update_timestep(ignore_custom_adaptation=True)
    if "niter0" in integrator.solver_dict.keys():
        if integrator.solver_dict['niter0'] != 0 and integrator.solver_dict['niter1'] != 0:
            Tk0, CTk0 = D.ar_numpy.log(integrator.solver_dict['tau0']), math.log(integrator.solver_dict['niter0'])
            Tk1, CTk1 = D.ar_numpy.log(integrator.solver_dict['tau1']), math.log(integrator.solver_dict['niter1'])
            dnCTk = D.ar_numpy.asarray(CTk1 - CTk0, **integrator.array_constructor_kwargs)
            ddCTk = D.ar_numpy.asarray(Tk1 - Tk0, **integrator.array_constructor_kwargs)
            if ddCTk > 0:
                dCTk = dnCTk / ddCTk
            else:
                dCTk = D.ar_numpy.zeros_like(integrator.solver_dict['timestep'])
            tau2 = D.ar_numpy.exp(-safety_factor * dCTk)
        else:
            tau2 = D.numpy.inf
        total_error_tolerance = integrator.solver_dict['atol'] + integrator.solver_dict['rtol']
        with D.numpy.errstate(divide='ignore'):
            epsilon_current = D.ar_numpy.reciprocal(D.ar_numpy.linalg.norm(integrator.solver_dict['newton_prec1'] / total_error_tolerance))
        tau3 = D.ar_numpy.where(epsilon_current > 0.0, epsilon_current ** (1.0 / integrator.solver_dict['order']), 1.0)
        if integrator.solver_dict['newton_prec0'] > 0.0:
            with D.numpy.errstate(divide='ignore'):
                epsilon_last = D.ar_numpy.reciprocal(D.ar_numpy.linalg.norm(integrator.solver_dict['newton_prec0'] / total_error_tolerance))
            tau3 = tau3*D.ar_numpy.where(epsilon_last > 0.0, epsilon_last ** (1.0 / integrator.solver_dict['order']), 1.0)
        tau = (1 + D.ar_numpy.arctan(D.ar_numpy.minimum(tau2, tau3) - 1))
        return D.ar_numpy.minimum(tau * timestep, timestep_from_error), redo_step
    else:
        return timestep_from_error, redo_step
