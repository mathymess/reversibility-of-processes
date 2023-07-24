import torch
import torch.nn as nn

import numpy.typing
from typing import Tuple

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from generate_time_series import load_logistic_map_time_series
from datasets import chop_time_series_into_chunks, split_chunks_into_windows_and_targets


NDArray = numpy.typing.NDArray[numpy.floating]


class BayesianThreeFCLayers(PyroModule):
    def __init__(self, window_len: int, datapoint_size: int = 1, target_len: int = 1,
                 hidden_layer1_size: int = 10, hidden_layer2_size: int = 10,
                 prior_scale: float = 10.):
        super().__init__()
        if datapoint_size != 1 or target_len != 1:
            raise NotImplementedError("I didn't figure out how to support "
                                      "tensors of sophisticated shape")

        self.window_len: int = window_len
        self.datapoint_size: int = datapoint_size
        self.target_len: int = target_len

        self.hidden_layer1_size = hidden_layer1_size
        self.hidden_layer2_size = hidden_layer2_size

        self._init_layers(prior_scale)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=device)

    def _init_layers(self, prior_scale: float):
        size1 = (self.window_len * self.datapoint_size, self.hidden_layer1_size)
        size2 = (self.hidden_layer1_size, self.hidden_layer2_size)
        size3 = (self.hidden_layer2_size, self.target_len * self.datapoint_size)

        self.fc1 = PyroModule[nn.Linear](*size1)
        self.relu1 = nn.ReLU()
        self.fc2 = PyroModule[nn.Linear](*size2)
        self.relu2 = nn.ReLU()
        self.fc3 = PyroModule[nn.Linear](*size3)

        # Set layer parameters as random variables
        self.fc1.weight = PyroSample(dist.Normal(0., prior_scale).expand(size1[::-1]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., prior_scale).expand(size1[-1:]).to_event(1))
        self.fc2.weight = PyroSample(dist.Normal(0., prior_scale).expand(size2[::-1]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., prior_scale).expand(size2[-1:]).to_event(1))
        self.fc3.weight = PyroSample(dist.Normal(0., prior_scale).expand(size3[::-1]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(0., prior_scale).expand(size3[-1:]).to_event(1))

    def forward(self, windows: torch.tensor, y: torch.tensor = None) -> torch.tensor:
        # assert windows.shape[-1] == self.window_len * self.datapoint_size
        sigma = pyro.sample("sigma", dist.Uniform(0., 5.))

        ret = windows
        ret = self.relu1(self.fc1(ret))
        ret = self.relu2(self.fc2(ret))
        ret = self.fc3(ret)
        ret = ret.squeeze(-1)

        # Just copying this from the tutorial, I have no idea what it does.
        # https://pyro.ai/examples/bayesian_regression.html
        with pyro.plate("data", windows.shape[0]):
            obs = pyro.sample("obs", dist.Normal(ret, sigma), obs=y)

        return ret


def test_model_output_dimensions() -> None:
    model = BayesianThreeFCLayers(window_len=2, datapoint_size=1, target_len=1)
    batch = torch.tensor([[1., 2.], [7., 8.]])
    targets = model(batch)
    assert targets.ndim == 1 and targets.shape == (2,)
    print("test_model_output_dimensions passed successfully!")


def prepare_simple_1d_time_series(ts: NDArray,
                                  window_len: int,
                                  take_each_nth_chunk: int = 1,
                                  reverse: bool = False) -> Tuple[NDArray, NDArray]:
    chunks = chop_time_series_into_chunks(time_series=ts,
                                          chunk_len=window_len + 1,
                                          take_each_nth_chunk=take_each_nth_chunk,
                                          reverse=reverse)
    windows, targets = split_chunks_into_windows_and_targets(chunks, target_len=1)

    windows = windows.squeeze(-1)
    targets = targets.squeeze(-1)
    
    windows = torch.from_numpy(windows).float()
    targets = torch.from_numpy(targets).float()
    return windows, targets


def train_logistic():
    windows, targets = prepare_simple_1d_time_series(ts=load_logistic_map_time_series(length=2000),
                                                     window_len=1)
    model = BayesianThreeFCLayers(window_len=1, target_len=1, datapoint_size=1)

    mcmc = pyro.infer.MCMC(pyro.infer.NUTS(model, jit_compile=True), num_samples=100)
    mcmc.run(windows, targets)

    predictive = pyro.infer.Predictive(model=model, posterior_samples=mcmc.get_samples())

    x_test = torch.linspace(0.001, 0.999, 1000).reshape(-1, 1)
    y_test = (4.0 * x_test * (1 - x_test)).squeeze(-1)

    preds = predictive(x_test)
    print('preds["obs"].shape =', preds["obs"].shape)
    torch.save(preds["obs"], "20230723_preds.torch")


if __name__ == "__main__":
    # test_model_output_dimensions()
    train_logistic()
