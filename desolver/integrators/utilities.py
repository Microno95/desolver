import math
from desolver import backend as D
from desolver.integrators.integrator_types import TableauIntegrator

def implicit_aware_update_timestep(integrator: TableauIntegrator):
    timestep_from_error, redo_step = integrator.update_timestep(ignore_custom_adaptation=True)
    if "niter0" in integrator.solver_dict.keys():
        # Adjust the timestep according to the computational cost of 
        # solving the nonlinear system at each timestep
        # Based on the implementation here: https://www.sciencedirect.com/science/article/pii/S0168927418301387
        if integrator.solver_dict['niter0'] != 0 and integrator.solver_dict['niter1'] != 0:
            Tk0, CTk0 = D.ar_numpy.log(integrator.solver_dict['tau0']), math.log(integrator.solver_dict['niter0'])
            Tk1, CTk1 = D.ar_numpy.log(integrator.solver_dict['tau1']), math.log(integrator.solver_dict['niter1'])
            dnCTk = CTk1 - CTk0
            ddCTk = Tk1 - Tk0
            if ddCTk > 0:
                dCTk = dnCTk / ddCTk
                c_alpha, c_beta, c_lambda, c_delta = 1.19735982, 0.44611854, 1.38440318, 0.73715227
                c_s = D.ar_numpy.exp(-c_alpha*D.ar_numpy.tanh(c_beta*dCTk))
                tau2 = D.ar_numpy.where(
                    (1 <= c_s) & (c_s < c_lambda),
                    c_lambda,
                    D.ar_numpy.where(
                        (c_delta <= c_s) & (c_s < 1),
                        c_delta,
                        c_s
                    )
                )
            else:
                dCTk = D.ar_numpy.zeros_like(integrator.solver_dict['timestep'])
                tau2 = D.ar_numpy.ones_like(integrator.solver_dict['timestep'])*10.0
        else:
            tau2 = D.ar_numpy.ones_like(integrator.solver_dict['timestep'])
        # # ---- #
        # # Adjust the timestep according to the precision achieved by the 
        # # nonlinear system solver at each timestep
        # tau3 = D.ar_numpy.ones_like(integrator.solver_dict['timestep'])
        # if integrator.solver_dict['newton_prec1'] > 0.0:
        #     with D.numpy.errstate(divide='ignore'):
        #         epsilon_current = integrator.solver_dict['newton_tol'] / integrator.solver_dict['newton_prec1']
        #     tau3 = tau3*D.ar_numpy.where(D.ar_numpy.isfinite(epsilon_current), epsilon_current, 1.0)
        # else:
        #     tau3 = tau3*10.0
        # tau3 = D.ar_numpy.clip(tau3, min=0.8, max=10.0)
        # # ---- #
        # if tau2 is None:
        #     tau = tau3
        # else:
        #     tau = D.ar_numpy.minimum(tau2, tau3)
        corr = timestep_from_error/integrator.solver_dict["tau1"]
        tau = D.ar_numpy.sqrt(corr*(1 + D.ar_numpy.arctan((tau2 - 1))))
        return tau * integrator.solver_dict["tau1"], redo_step
    else:
        return timestep_from_error, redo_step
