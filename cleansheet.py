import torch
import torch.nn as nn
from tqdm.auto import trange
import numpy as np
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.contrib import forecast as pyro_metric

from bayesian import BayesianThreeFCLayers, plot_predictions


class BNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=10, n_hid_layers=5, prior_scale=5.):
        super().__init__()

        self.activation = nn.Tanh()  # could also be ReLU or LeakyReLU
        assert in_dim > 0 and out_dim > 0 and hid_dim > 0 and n_hid_layers > 0  # make sure the dimensions are valid

        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.activation(self.layers[0](x))  # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # hidden --> hidden
        mu = self.layers[-1](x).squeeze()  # hidden --> output
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # infer the response noise

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu


x_train = torch.linspace(0., 1., 200)
y_train = 3.0 * x_train + 0.2 * torch.rand(x_train.shape)
# plt.scatter(x_train, y_train)
# plt.show()

# model = BNN(hid_dim=10, n_hid_layers=3, prior_scale=2.)
x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
model = BayesianThreeFCLayers(hidden_size=10, window_len=1, prior_scale=0.5)

mean_field_guide = AutoDiagonalNormal(model)
optimizer = pyro.optim.Adam({"lr": 0.01})

svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())
pyro.clear_param_store()

num_epochs = 25000
progress_bar = trange(num_epochs)

losses = []
rmse_losses = []
for epoch in progress_bar:
    loss = svi.step(x_train, y_train)
    losses.append(loss)
    progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")

    if epoch % 1000 == 0:
        predictive = pyro.infer.Predictive(model=model, guide=mean_field_guide,
                                           num_samples=100)
        preds = predictive(x_train)["obs"]
        rmse = pyro_metric.eval_rmse(preds, y_train)
        rmse_losses.append(rmse)

        plot_predictions(true=y_train, pred_mean=preds.mean(0), pred_std=preds.std(0))
        plt.show()

plt.plot(losses)
plt.show()

plt.plot(rmse_losses)
plt.show()
