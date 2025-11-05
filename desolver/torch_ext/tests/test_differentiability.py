import pytest
try:
    import torch
    from desolver.torch_ext import torch_solve_ivp
    pytorch_available = True
except ImportError:
    pytorch_available = False


def rhs(t, state, k, m):
    return torch.stack([state[...,1], -k/m*state[...,0]], dim=-1)


@pytest.mark.slow
def test_gradcorrectness_variable_steps(pytorch_only):
    constants = dict(
        k = 1.0,
        m = 1.0
    )

    T = 2*torch.pi*(constants['m']/constants['k'])**0.5

    y_init = torch.tensor([1., 0.], dtype=torch.float64)
    
    def test_fn(y, initial_time, final_time, spring_constant, mass_constant):
        res_out = torch_solve_ivp(rhs, t_span=(initial_time, final_time), y0=y, method="RK87", args=[spring_constant, mass_constant], atol=1e-10, rtol=1e-10)
        return res_out.y[...,-1].sin().abs().mean() + res_out.t[-1].square().sum() + res_out.t[0].square().sum()

    grad_inputs = [y_init.clone().requires_grad_(True), torch.tensor(0.0, dtype=torch.float64, requires_grad=True), torch.tensor(T/3, dtype=torch.float64, requires_grad=True),
                                torch.tensor(constants['k'], dtype=torch.float64, requires_grad=True), torch.tensor(constants['m'], dtype=torch.float64, requires_grad=True)]
    gradgrad_inputs = torch.tensor(0.2+1/3)

    assert torch.autograd.gradcheck(test_fn, grad_inputs, check_forward_ad=True, check_backward_ad=True, raise_exception=True)
    # assert torch.autograd.gradgradcheck(test_fn, grad_inputs, gradgrad_inputs, check_fwd_over_rev=True, check_rev_over_rev=True, raise_exception=True)


@pytest.mark.slow
def test_gradcorrectness_fixed_steps(pytorch_only):
    constants = dict(
        k = 1.0,
        m = 1.0
    )

    T = 2*torch.pi*(constants['m']/constants['k'])**0.5

    t0 = 0.0
    
    y_init = torch.tensor([1., 0.], dtype=torch.float64)
    
    def test_fn(y):
        res_out = torch_solve_ivp(rhs, t_span=(t0, T/3+T/5), y0=y, method="RK5Solver", kwargs=constants, first_step=(T/3+T/7)/24)
        return res_out.y[0,1].abs().mean()

    assert torch.autograd.gradcheck(test_fn, y_init.clone().requires_grad_(True), atol=1e-4, rtol=1e-4, check_forward_ad=True, check_backward_ad=True, raise_exception=True)
    # assert torch.autograd.gradgradcheck(test_fn, y_init.clone().requires_grad_(True), torch.ones_like(y_init).square().mean().requires_grad_(True)*0.182, atol=1e-4, rtol=1e-4, check_fwd_over_rev=True, check_rev_over_rev=True, raise_exception=True)


@pytest.mark.slow
def test_gradcorrectness_multiple_fixed_steps(pytorch_only):
    constants = dict(
        k = 1.0,
        m = 1.0
    )

    T = 2*torch.pi*(constants['m']/constants['k'])**0.5

    y_init = torch.tensor([1., 0.], dtype=torch.float64)
    
    def test_fn(y, spring_constant, mass_constant):
        res_out = torch_solve_ivp(rhs, t_span=(0.0, T/3+T/7), y0=y, method="RK5Solver", args=[spring_constant, mass_constant], first_step=(T/3+T/7)/24)
        return res_out.y[...,[4, 18, -1]].sin().abs().mean() + res_out.t[-1].square().sum() + res_out.t[0].square().sum()

    grad_inputs = [y_init.clone().requires_grad_(True), torch.tensor(constants['k'], dtype=torch.float64, requires_grad=True), torch.tensor(constants['m'], dtype=torch.float64, requires_grad=True)]
    gradgrad_inputs = torch.tensor(1.0+1/3)

    assert torch.autograd.gradcheck(test_fn, grad_inputs, check_forward_ad=True, check_backward_ad=True, raise_exception=True)
    # assert torch.autograd.gradgradcheck(test_fn, grad_inputs, gradgrad_inputs, check_fwd_over_rev=True, check_rev_over_rev=True, raise_exception=True)
