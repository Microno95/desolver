import os

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['DES_BACKEND']='torch'

import pandas as pd
import desolver as de
import desolver.backend as D
import tqdm.auto as tqdm

for ffmt in D.available_float_fmt():
    D.set_float_fmt(ffmt)
    
    print("Testing {} float format".format(D.float_fmt()))

    import torch

    torch.set_printoptions(precision=17)
    torch.set_num_threads(1)

    torch.autograd.set_detect_anomaly(True)

    class NNController(torch.nn.Module):

        def __init__(self, in_dim=2, out_dim=2, inter_dim=50, append_time=False):
            super().__init__()

            self.append_time = append_time

            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim+(1 if append_time else 0), inter_dim),
                torch.nn.Softplus(),
                torch.nn.Linear(inter_dim, out_dim),
                torch.nn.Sigmoid()
            )

            for idx,m in enumerate(self.net.modules()):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight, gain=1.0)
                    torch.nn.init.constant_(m.bias, 0.0)

        def forward(self, t, y, dy):
            if self.append_time:
                return self.net(torch.cat([y.view(-1), dy.view(-1), t.view(-1)]))
            else:
                return self.net(torch.cat([y, dy]))

    class SimpleODE(torch.nn.Module):
        def __init__(self, inter_dim=2, k=1.0):
            super().__init__()
            self.nn_controller = NNController(in_dim=4, out_dim=1, inter_dim=inter_dim)
            self.A = torch.tensor([[0.0, 1.0],[-k, 0.0]], requires_grad=True)

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

            return dy + torch.cat([torch.tensor([0.0]), controller_effect * 2.0 - 1.0])

    with de.BlockTimer(section_label="Integrator Tests"):
        for i in sorted(set(de.available_methods.values()), key=lambda x:x.__name__):
            try:
                yi1 = torch.tensor([1.0, 1.0], requires_grad=True)
                df  = SimpleODE(k=5.0)

                a = de.OdeSystem(df, yi1, t=(0, 10), dt=0.5, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5)
                a.set_method(i)
                a.integrate(eta=True)

                dyfdyi = D.jacobian(a.y[-1], a.y[0])
                dyi = D.array([0.0, 1.0]) * 1e-5
                dyf = D.einsum("nk,k->n", dyfdyi, dyi)
                yi2 = yi1 + dyi

                b = de.OdeSystem(df, yi2, t=(0, 10), dt=0.5, rtol=D.epsilon()**0.5, atol=D.epsilon()**0.5)
                b.set_method(i)
                b.integrate(eta=True)

                true_diff = b.y[-1] - a.y[-1]

                print(D.norm(true_diff - dyf), D.epsilon()**0.5)

                assert(D.allclose(true_diff, dyf, rtol=100 * D.epsilon()**0.5, atol=D.epsilon()**0.5))
                print("{} method test succeeded!".format(a.integrator))
            except:
                raise RuntimeError("Test failed for integration method: {}".format(a.integrator))
        print("")

    print("{} backend test passed successfully!".format(os.environ['DES_BACKEND']))    
