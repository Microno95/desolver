import typing
import desolver.backend as D

def compute_step(rhs: typing.Callable, initial_time, initial_state, timestep, intermediate_stages_in, intermediate_stages_out, rk_tableau, additional_kwargs=dict()):
    r"""Computes a single iteration of a generic Runge-Kutta stage calculation

    A generic RK stage calculation involves taking each of the tableau coefficients of each stage,
    computing the product-sum over each of the stages, calculating the derivative, and then storing
    this value in the stage calculation.
    
    In an explicit scheme, `intermediate_stages_in` and `intermediate_stages_out` are dependent on
    each other as the results of the `i+1`-th stage depends on the `i`-th stage whereas in an
    implicit scheme these can be computed independently as an external solver will resolve the values.

    Parameters
    ----------
    rhs : Callable
        The right-hand side of the differential equation, must take an array-like `state`,
        a scalar `time`, and may additionally take extra args/kwargs that are parameters
        of the derivative.
    initial_state : array-like, (...)
        An arbitrary array-like describing the state of the ODE system
    time : float
        A scalar time denoting the time of the ODE system
    timestep : float
        A scalar timestep denoting the timestep of the integrator
    intermediate_stages_in, intermediate_stages_out : array-like, (N,...)
        Array-like storage for the intermediate stage values for the input and output
        respectively. While more verbose, having a second argument for the output
        allows reusing storage instead of reallocating.
    rk_tableau : array-like
        The Runge-Kutta Butcher tableau denoting the stage coefficients. The first column
        is the timestep coefficients (c_i) and the remaining N columns are the stage 
        coefficients (a_i).
    *additional_args, **additional_kwargs : iterable, dict
        Additional arguments to pass to `rhs`

    Returns
    -------
    intermediate_stages_out : array-like
        Returns the computed stage values of the input RK Tableau.
    """
    
    for stage in range(intermediate_stages_in.shape[-1]):
        intermediate_dstate = timestep * D.ar_numpy.sum(intermediate_stages_in * rk_tableau[stage, 1:], axis=-1)
        intermediate_rhs = rhs(
            initial_time + timestep * rk_tableau[stage, 0], 
            initial_state + intermediate_dstate,
            **additional_kwargs
        )
        intermediate_stages_out[...,stage] = intermediate_rhs
    return intermediate_stages_out, intermediate_dstate, intermediate_rhs
