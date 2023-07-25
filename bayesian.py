import os
import numpy as np
import numpy.typing
from typing import Tuple
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from generate_time_series import load_logistic_map_time_series, load_garch_time_series
from datasets import chop_time_series_into_chunks, split_chunks_into_windows_and_targets


NDArray = numpy.typing.NDArray[numpy.floating]


class BayesianThreeFCLayers(PyroModule):
    def __init__(self, window_len: int, datapoint_size: int = 1, target_len: int = 1,
                 hidden_size: int = 10, prior_scale: float = 10.):
        if datapoint_size != 1 or target_len != 1:
            raise NotImplementedError("I didn't figure out how to support "
                                      "target tensors of sophisticated shape")
        super().__init__()

        self.input_size: int = window_len * datapoint_size
        self.hidden_size: int = hidden_size
        self.output_size: int = target_len * datapoint_size

        self._init_layers(prior_scale)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device=device)

    def _init_layers(self, prior_scale: float):
        size1 = (self.input_size, self.hidden_size)
        size2 = (self.hidden_size, self.hidden_size)
        size3 = (self.hidden_size, self.output_size)

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
        assert windows.ndim in (1, 2)
        assert windows.shape[-1] == self.input_size

        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # Infer the response noise

        ret = windows
        ret = self.relu1(self.fc1(ret))
        ret = self.relu2(self.fc2(ret))
        ret = self.fc3(ret)

        # Just copying this from the tutorial, I have no idea what it does.
        # https://pyro.ai/examples/bayesian_regression.html
        with pyro.plate("data", windows.shape[0]):
            # TODO: .squeeze(-1) is a hack, but otherwise everything fails if y or ret are not 1D
            obs = pyro.sample("obs", dist.Normal(ret.squeeze(-1), sigma * sigma),
                              obs=None if y is None else y.squeeze(-1))

        return ret


def test_model_output_dimensions() -> None:
    model = BayesianThreeFCLayers(window_len=2, datapoint_size=1, target_len=1)
    batch = torch.tensor([[1., 2.], [3., 4.]])
    targets = model(batch)
    assert targets.ndim == 2 and targets.shape == (2, 1), f"targets.shape={targets.shape}"
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


def get_samples_from_posterior_predictive(windows: torch.tensor,
                                          targets: torch.tensor,
                                          num_samples: int = 100) -> pyro.infer.Predictive:
    assert windows.ndim == targets.ndim == 2
    assert windows.shape[0] == targets.shape[0]

    model = BayesianThreeFCLayers(window_len=windows.shape[-1], target_len=1, datapoint_size=1)
    mcmc = pyro.infer.MCMC(pyro.infer.NUTS(model, jit_compile=True), num_samples)
    mcmc.run(windows, targets)
    predictive = pyro.infer.Predictive(model=model, posterior_samples=mcmc.get_samples())

    return predictive


def posterior_predictive_forward_and_backward(
        train_ts: NDArray,
        test_ts: NDArray,
        save_dir: str,
        window_len: int = 1,
        num_samples: int = 100) -> Tuple[torch.tensor, torch.tensor]:
    os.makedirs(save_dir, exist_ok=False)

    predictive_f = get_samples_from_posterior_predictive(
            *prepare_simple_1d_time_series(train_ts, window_len), num_samples)
    predictive_b = get_samples_from_posterior_predictive(
            *prepare_simple_1d_time_series(train_ts, window_len, reverse=True), num_samples)

    windows_test, targets_test = prepare_simple_1d_time_series(test_ts, window_len)

    samples_f = predictive_f(windows_test)
    samples_b = predictive_b(windows_test)

    torch.save(windows_test, os.path.join(save_dir, "windows_test.torch"))
    torch.save(targets_test, os.path.join(save_dir, "targets_test.torch"))
    torch.save(samples_f, os.path.join(save_dir, "samples.forward.torch"))
    torch.save(samples_b, os.path.join(save_dir, "samples.backward.torch"))

    return samples_f, samples_b


def train_logistic():
    posterior_predictive_forward_and_backward(
        train_ts=load_logistic_map_time_series(2000),
        test_ts=np.linspace(0.001, 0.999, 100).reshape(-1, 1),
        save_dir="20230724_preds/logistics1",
        window_len=1)

    posterior_predictive_forward_and_backward(
        train_ts=load_logistic_map_time_series(2000),
        test_ts=np.linspace(0.001, 0.999, 100).reshape(-1, 1),
        save_dir="20230724_preds/logistics2",
        window_len=2)


def train_garch():
    torch.manual_seed(42)
    test_ts = numpy.random.uniform(0, 1, size=(1000, 1))

    # posterior_predictive_forward_and_backward(
    #     train_ts=load_garch_time_series(2000, coef_alpha=0.1),
    #     test_ts=0.3 * test_ts,
    #     save_dir="20230724_preds/garch01",
    #     window_len=3)

    posterior_predictive_forward_and_backward(
        train_ts=load_garch_time_series(2000, coef_alpha=0.7),
        test_ts=2. * test_ts,
        save_dir="20230724_preds/garch02",
        window_len=3,
        num_samples=20)


if __name__ == "__main__":
    test_model_output_dimensions()
    # train_garch()
    # train_logistic()
