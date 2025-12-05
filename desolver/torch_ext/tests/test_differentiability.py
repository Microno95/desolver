import pytest
import desolver as de
try:
    import torch
    from desolver.torch_ext import torch_solve_ivp
    pytorch_available = True
except ImportError:
    pytorch_available = False


def rhs_harmonic_oscillator(t, state, k, m):
    return torch.stack([state[...,1], -k/m*state[...,0]], dim=-1)


def rhs_rossler(t, state, a, b, c):
    return torch.stack([
        -state[...,1] - state[...,2],
         state[...,0]+a*state[...,1],
         b + state[...,2]*(state[...,0] - c)
    ], dim=-1)


@pytest.mark.slow
@pytest.mark.parametrize("integrator_to_test", [de.integrators.RK8713MSolver, de.integrators.RadauIIA5])
def test_gradcorrectness_variable_steps(pytorch_only, integrator_to_test):
    constants = dict(
        a = 0.2,
        b = 0.2,
        c = 5.7,
    )

    if integrator_to_test == de.integrators.RadauIIA5:
        T = torch.pi/20
    else:
        T = 2*torch.pi

    y_init = torch.tensor([1., 0., 0.], dtype=torch.float64)
    print()
    def test_fn(y, initial_time, final_time, a_constant, b_constant, c_constant):
        res_out = torch_solve_ivp(rhs_rossler, t_span=(initial_time, final_time), y0=y, method=integrator_to_test, args=[a_constant, b_constant, c_constant], atol=1e-10, rtol=1e-10, show_prog_bar=False)
        return res_out.y[...,-1].sin().abs().mean() + res_out.t[-1].square().sum() + res_out.t[0].square().sum()

    grad_inputs = [y_init.clone().requires_grad_(True), torch.tensor(0.0, dtype=torch.float64, requires_grad=True), torch.tensor(T/3, dtype=torch.float64, requires_grad=True),
                   torch.tensor(constants['a'], dtype=torch.float64, requires_grad=True), torch.tensor(constants['b'], dtype=torch.float64, requires_grad=True),
                   torch.tensor(constants['c'], dtype=torch.float64, requires_grad=True)]
    gradgrad_inputs = torch.tensor(0.2+1/3)

    torch.autograd.gradcheck(test_fn, grad_inputs, check_forward_ad=False, check_backward_ad=True, atol=1e-3, rtol=1e-2, raise_exception=True)
    print("First order reverse-AD passed")
    torch.autograd.gradcheck(test_fn, grad_inputs, check_forward_ad=True, check_backward_ad=False, atol=1e-3, rtol=1e-2, raise_exception=True)
    print("First order forward-AD passed")
    torch.autograd.gradgradcheck(test_fn, grad_inputs, gradgrad_inputs, check_fwd_over_rev=False, check_rev_over_rev=True, atol=1e-2, rtol=1e-1, check_undefined_grad=True, raise_exception=True)
    torch.autograd.gradgradcheck(test_fn, grad_inputs, gradgrad_inputs, check_fwd_over_rev=True, check_rev_over_rev=False, atol=1e-2, rtol=1e-1, check_undefined_grad=False, raise_exception=True)
    print("Second order AD passed")


@pytest.mark.slow
def test_gradcorrectness_fixed_steps(pytorch_only):
    constants = dict(
        k = 1.0,
        m = 1.0
    )

    T = 2*torch.pi*(constants['m']/constants['k'])**0.5

    y_init = torch.tensor([1., 0.], dtype=torch.float64)
    print()
    def test_fn(y, initial_time, final_time, spring_constant, mass_constant):
        res_out = torch_solve_ivp(rhs_harmonic_oscillator, t_span=(initial_time, final_time), y0=y, method=de.integrators.RK5Solver, args=[spring_constant, mass_constant], first_step=(final_time - initial_time).detach()/64, show_prog_bar=True)
        return res_out.y[0,-1].abs().mean()

    grad_inputs = [y_init.clone().requires_grad_(True), torch.tensor(0.0, dtype=torch.float64, requires_grad=True), torch.tensor(T/3, dtype=torch.float64, requires_grad=True),
                                torch.tensor(constants['k'], dtype=torch.float64, requires_grad=True), torch.tensor(constants['m'], dtype=torch.float64, requires_grad=True)]
    gradgrad_inputs = torch.tensor(0.2+1/3)

    torch.autograd.gradcheck(test_fn, grad_inputs, check_forward_ad=False, check_backward_ad=True, raise_exception=True)
    torch.autograd.gradcheck(test_fn, grad_inputs, check_forward_ad=True, check_backward_ad=False, raise_exception=True)
    torch.autograd.gradgradcheck(test_fn, grad_inputs, gradgrad_inputs, check_fwd_over_rev=False, check_rev_over_rev=True, atol=1e-2, rtol=1e-1, check_undefined_grad=True, raise_exception=True)
    torch.autograd.gradgradcheck(test_fn, grad_inputs, gradgrad_inputs, check_fwd_over_rev=True, check_rev_over_rev=False, atol=1e-2, rtol=1e-1, check_undefined_grad=False, raise_exception=True)


@pytest.mark.slow
def test_gradcorrectness_multiple_fixed_steps(pytorch_only):
    constants = dict(
        k = 1.0,
        m = 1.0
    )

    T = 2*torch.pi*(constants['m']/constants['k'])**0.5

    y_init = torch.tensor([1., 0.], dtype=torch.float64)
    print()
    def test_fn(y, spring_constant, mass_constant):
        res_out = torch_solve_ivp(rhs_harmonic_oscillator, t_span=(0.0, T/3+T/7), y0=y, method=de.integrators.RK5Solver, args=[spring_constant, mass_constant], first_step=(T/3+T/7)/24, show_prog_bar=True)
        return res_out.y[...,[4, 18, -1]].sin().abs().mean() + res_out.t[-1].square().sum() + res_out.t[0].square().sum()

    grad_inputs = [y_init.clone().requires_grad_(True), torch.tensor(constants['k'], dtype=torch.float64, requires_grad=True), torch.tensor(constants['m'], dtype=torch.float64, requires_grad=True)]
    gradgrad_inputs = torch.tensor(0.2+1/3)

    torch.autograd.gradcheck(test_fn, grad_inputs, check_forward_ad=False, check_backward_ad=True, raise_exception=True)
    torch.autograd.gradcheck(test_fn, grad_inputs, check_forward_ad=True, check_backward_ad=False, raise_exception=True)
    torch.autograd.gradgradcheck(test_fn, grad_inputs, gradgrad_inputs, check_fwd_over_rev=False, check_rev_over_rev=True, atol=1e-2, rtol=1e-1, check_undefined_grad=True, raise_exception=True)
    torch.autograd.gradgradcheck(test_fn, grad_inputs, gradgrad_inputs, check_fwd_over_rev=True, check_rev_over_rev=False, atol=1e-2, rtol=1e-1, check_undefined_grad=False, raise_exception=True)
