import pytest
import desolver as de
import desolver.backend as D
import numpy as np


@pytest.mark.skipif(D.backend() != 'torch', reason="PyTorch Unavailable")
@pytest.mark.parametrize('ffmt', D.available_float_fmt())
@pytest.mark.parametrize('integrator_name', sorted(set(de.available_methods(False).values()), key=lambda x: x.__name__))
def test_gradients(ffmt, integrator_name):
    D.set_float_fmt(ffmt)

    print("Testing {} float format".format(D.float_fmt()))

    import torch

    torch.set_printoptions(precision=17)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    with de.utilities.BlockTimer(section_label="Integrator Tests"):
        yi1 = D.array([1.0, 0.0], requires_grad=True).to(device)
        df = SimpleODE(k=1.0)

        a = de.OdeSystem(df, yi1, t=(0, 1.), dt=0.0675, rtol=D.epsilon() ** 0.5, atol=D.epsilon() ** 0.5)
        a.set_method(integrator_name)
        a.integrate(eta=False)

        dyfdyi = D.jacobian(a.y[-1], a.y[0])
        dyi = D.array([0.0, 1.0]).to(device) * D.epsilon() ** 0.5
        dyf = D.einsum("nk,k->n", dyfdyi, dyi)
        yi2 = yi1 + dyi

        print(a.y[-1].device)

        b = de.OdeSystem(df, yi2, t=(0, 1.), dt=0.0675, rtol=D.epsilon() ** 0.5, atol=D.epsilon() ** 0.5)
        b.set_method(integrator_name)
        b.integrate(eta=False)

        true_diff = b.y[-1] - a.y[-1]

        print(D.norm(true_diff - dyf), D.epsilon() ** 0.5)

        assert (D.allclose(true_diff, dyf, rtol=4 * D.epsilon() ** 0.5, atol=4 * D.epsilon() ** 0.5))
        print("{} method test succeeded!".format(a.integrator))
        print("")

    print("{} backend test passed successfully!".format(D.backend()))


if __name__ == "__main__":
    np.testing.run_module_suite()
