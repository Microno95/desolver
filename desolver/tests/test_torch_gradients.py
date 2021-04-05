import pytest
import desolver as de
import desolver.backend as D
import numpy as np

integrator_set = set(de.available_methods(False).values())
integrator_set = sorted(integrator_set, key=lambda x: x.__name__)
explicit_integrator_set = [
    pytest.param(intg, marks=pytest.mark.explicit) for intg in integrator_set if not intg.__implicit__
]
implicit_integrator_set = [
    pytest.param(intg, marks=pytest.mark.implicit) for intg in integrator_set if intg.__implicit__
]


devices_set = ['cpu']
    
@pytest.mark.torch_gradients
@pytest.mark.skip(reason="Test too slow, needs refactoring")
@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
@pytest.mark.parametrize('use_richardson_extrapolation', [False])
@pytest.mark.parametrize('device', devices_set)
def test_gradients_simple_decay(ffmt, integrator, use_richardson_extrapolation, device):
    D.set_float_fmt(ffmt)
    if integrator.__symplectic__:
        pytest.skip("Exponential decay system is not in the form compatible with symplectic integrators")
    if integrator.__implicit__ and use_richardson_extrapolation:
        pytest.skip("Implicit methods are unstable with richardson extrapolation")
    print("Testing {} float format".format(D.float_fmt()))

    import torch

    torch.set_printoptions(precision=17)
    
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    device = torch.device(device)

    torch.autograd.set_detect_anomaly(True)

    def rhs(t, state, k, **kwargs):
        return -k*state

    y_init = D.array(5.0, requires_grad=True)
    csts   = dict(k=1.0)
    dt     = (D.epsilon() ** 0.5)**(1.0/(1+integrator.order))
    
    def true_solution_decay(t, initial_state, k):
        return initial_state * D.exp(-k*t)


    method = integrator
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        
    with de.utilities.BlockTimer(section_label="Integrator Tests"):
        y_init = D.ones((5,5,2), requires_grad=True).to(device)
        y_init = y_init * 5.

        a = de.OdeSystem(rhs, y_init, t=(0, 0.0025), dt=1e-5, rtol=D.epsilon() ** 0.5, atol=D.epsilon() ** 0.5, constants=csts)
        a.set_method(method)
        if a.integrator.__implicit__:
            a.rtol = a.atol = D.epsilon()**0.25
            
        a.integrate(eta=True)
        
        Jy = D.jacobian(a.y[-1], a.y[0])
        true_Jy = D.jacobian(true_solution_decay(a.t[-1], a.y[0], **csts), a.y[0])
        
        print(true_Jy, Jy, D.norm(true_Jy - Jy), D.epsilon() ** 0.5)

        assert (D.allclose(true_Jy, Jy, rtol=4 * a.rtol, atol=4 * a.atol))
        print("{} method test succeeded!".format(a.integrator))
        print("")

    print("{} backend test passed successfully!".format(D.backend()))

    
@pytest.mark.torch_gradients
@pytest.mark.skip(reason="Test too slow, needs refactoring")
@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
@pytest.mark.parametrize('use_richardson_extrapolation', [False])
@pytest.mark.parametrize('device', devices_set)
def test_gradients_simple_oscillator(ffmt, integrator, use_richardson_extrapolation, device):
    D.set_float_fmt(ffmt)
    if integrator.__implicit__ and use_richardson_extrapolation:
        pytest.skip("Implicit methods are unstable with richardson extrapolation")

    print("Testing {} float format".format(D.float_fmt()))

    import torch

    torch.set_printoptions(precision=17)
    
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    device = torch.device(device)

    torch.autograd.set_detect_anomaly(True)

    def rhs(t, state, k, m, **kwargs):
        return D.array([[0.0, 1.0], [-k/m,  0.0]], device=device)@state
    
    csts = dict(k=1.0, m=1.0)
    T    = 2*D.pi*D.sqrt(D.array(csts['m']/csts['k'])).to(device)
    dt   = (D.epsilon() ** 0.5)**(1.0/(1+integrator.order))
    
    def true_solution_sho(t, initial_state, k, m):
        w2 = D.array(k/m).to(device)
        w = D.sqrt(w2)
        A = D.sqrt(initial_state[0]**2 + initial_state[1]**2/w2)
        phi = D.atan2(-initial_state[1], w*initial_state[0])
        return D.stack([
            A * D.cos(w*t + phi),
            -w * A * D.sin(w*t + phi)
        ]).T


    method = integrator
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        
    with de.utilities.BlockTimer(section_label="Integrator Tests"):
        y_init = D.array([1., 1.], requires_grad=True).to(device)

        a = de.OdeSystem(rhs, y_init, t=(0, T/100), dt=1e-5, rtol=D.epsilon() ** 0.5, atol=D.epsilon() ** 0.5, constants=csts)
        a.set_method(method)
        if a.integrator.__implicit__:
            a.rtol = a.atol = D.epsilon()**0.25
        a.integrate(eta=True)
        
        Jy = D.jacobian(a.y[-1], a.y[0])
        true_Jy = D.jacobian(true_solution_sho(a.t[-1], a.y[0], **csts), a.y[0])
        
        print(D.norm(true_Jy - Jy), D.epsilon() ** 0.5)

        assert (D.allclose(true_Jy, Jy, rtol=4 * a.rtol, atol=4 * a.atol))
        print("{} method test succeeded!".format(a.integrator))
        print("")

    print("{} backend test passed successfully!".format(D.backend()))
    

@pytest.mark.torch_gradients
@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator', explicit_integrator_set + implicit_integrator_set)
@pytest.mark.parametrize('use_richardson_extrapolation', [False])
@pytest.mark.parametrize('device', devices_set)
def test_gradients_complex(ffmt, integrator, use_richardson_extrapolation, device):
    D.set_float_fmt(ffmt)
    if integrator.__implicit__ and use_richardson_extrapolation:
        pytest.skip("Implicit methods are unstable with richardson extrapolation")

    print("Testing {} float format".format(D.float_fmt()))

    import torch

    torch.set_printoptions(precision=17)
    
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    device = torch.device(device)

    torch.autograd.set_detect_anomaly(True)

    class NNController(torch.nn.Module):

        def __init__(self, in_dim=2, out_dim=2, inter_dim=50, append_time=False):
            super().__init__()

            self.append_time = append_time

            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim + (1 if append_time else 0), inter_dim),
                torch.nn.Softplus(),
                torch.nn.Linear(inter_dim, out_dim),
                torch.nn.Sigmoid()
            )

            for idx, m in enumerate(self.net.modules()):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                    torch.nn.init.constant_(m.bias, 0.0)

        def forward(self, t, y, dy):
            if self.append_time:
                return self.net(torch.cat([y.view(-1), dy.view(-1), t.view(-1)]))
            else:
                return self.net(torch.cat([y, dy]))

    class SimpleODE(torch.nn.Module):
        def __init__(self, inter_dim=10, k=1.0):
            super().__init__()
            self.nn_controller = NNController(in_dim=4, out_dim=1, inter_dim=inter_dim)
            self.A = torch.nn.Parameter(torch.tensor([[0.0, 1.0], [-k, -1.0]], requires_grad=False))

        def forward(self, t, y, params=None):
            if not isinstance(t, torch.Tensor):
                torch_t = torch.tensor(t)
            else:
                torch_t = t
            if not isinstance(y, torch.Tensor):
                torch_y = torch.tensor(y)
            else:
                torch_y = y
            if params is not None:
                if not isinstance(params, torch.Tensor):
                    torch_params = torch.tensor(params)
                else:
                    torch_params = params

            dy = torch.matmul(self.A, torch_y)

            controller_effect = self.nn_controller(torch_t, torch_y, dy) if params is None else params

            return dy + torch.cat([torch.tensor([0.0]).to(dy), (controller_effect * 2.0 - 1.0)])

    method = integrator
    if use_richardson_extrapolation:
        method = de.integrators.generate_richardson_integrator(method)
        
    with de.utilities.BlockTimer(section_label="Integrator Tests"):
        yi1 = D.array([1.0, 0.0], requires_grad=True).to(device)
        df = SimpleODE(k=1.0)

        a = de.OdeSystem(df, yi1, t=(0, 0.1), dt=0.025, rtol=D.epsilon() ** 0.5, atol=D.epsilon() ** 0.5)
        a.set_method(method)
        if a.integrator.__implicit__:
            a.rtol = a.atol = D.epsilon()**0.25
        a.integrate(eta=True)

        dyfdyi = D.jacobian(a.y[-1], a.y[0])
        dyi = D.array([0.0, 1.0]).to(device) * D.epsilon() ** 0.5
        dyf = D.einsum("nk,k->n", dyfdyi, dyi)
        yi2 = yi1 + dyi

        print(a.y[-1].device)

        b = de.OdeSystem(df, yi2, t=(0, 0.1), dt=0.025, rtol=a.rtol, atol=a.atol)
        b.set_method(method)
        b.integrate(eta=True)

        true_diff = b.y[-1] - a.y[-1]

        print(D.norm(true_diff - dyf), D.epsilon() ** 0.5)

        assert (D.allclose(true_diff, dyf, rtol=4 * a.rtol, atol=4 * a.atol))
        print("{} method test succeeded!".format(a.integrator))
        print("")

    print("{} backend test passed successfully!".format(D.backend()))
