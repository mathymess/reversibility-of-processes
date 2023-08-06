import os
import numpy as np
import numpy.typing
from typing import Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import Predictive

from generate_time_series import load_logistic_map_time_series, load_garch_time_series
from datasets import chop_time_series_into_chunks, split_chunks_into_windows_and_targets


NDArray = numpy.typing.NDArray[numpy.floating]


def prepare_simple_1d_time_series(ts: NDArray,
                                  window_len: int,
                                  take_each_nth_chunk: int = 1,
                                  reverse: bool = False) -> Tuple[torch.tensor, torch.tensor]:
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


class BayesTrainData:
    def __init__(self, ts: NDArray, window_len: int = 1, noise_std: float = 0.1):
        self.ts = ts

        self.noisy_ts = ts + np.random.normal(scale=noise_std, size=ts.shape)

        self.windows_f, self.targets_f = prepare_simple_1d_time_series(self.noisy_ts,
                                                                       window_len)
        self.windows_b, self.targets_b = prepare_simple_1d_time_series(self.noisy_ts,
                                                                       window_len,
                                                                       reverse=True)


def save_train_data_to_drive(train_d: BayesTrainData, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=False)
    torch.save(train_d.ts, os.path.join(save_dir, "ts.torch"))
    torch.save(train_d.noisy_ts, os.path.join(save_dir, "noisy_ts.torch"))
    torch.save(train_d.windows_f, os.path.join(save_dir, "windows_f.torch"))
    torch.save(train_d.targets_f, os.path.join(save_dir, "targets_f.torch"))
    torch.save(train_d.windows_b, os.path.join(save_dir, "windows_b.torch"))
    torch.save(train_d.targets_b, os.path.join(save_dir, "targets_b.torch"))


def plot_with_noise(ts: NDArray, noise_std: float = 1.):
    noisy_ts = ts + np.random.normal(scale=noise_std, size=ts.shape)
    plt.close()
    plt.plot(ts, "o", alpha=0.5)
    plt.plot(noisy_ts, "o")
    plt.grid()
    plt.show()


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

        obs = pyro.sample("obs",  # noqa: F841
                          dist.Normal(ret, sigma * sigma).to_event(1),
                          obs=y)

        return ret


def test_model_output_dimensions() -> None:
    model = BayesianThreeFCLayers(window_len=2, datapoint_size=1, target_len=1)
    batch = torch.tensor([[1., 2.], [3., 4.]])
    targets = model(batch)
    assert targets.ndim == 2 and targets.shape == (2, 1), f"targets.shape={targets.shape}"
    print("test_model_output_dimensions passed successfully!")


def get_samples_from_posterior_predictive(windows: torch.tensor,
                                          targets: torch.tensor,
                                          num_samples: int = 100) -> Predictive:
    assert windows.ndim == targets.ndim == 2
    assert windows.shape[0] == targets.shape[0]

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    model = BayesianThreeFCLayers(window_len=windows.shape[-1], target_len=1, datapoint_size=1)
    mcmc = pyro.infer.MCMC(pyro.infer.NUTS(model, jit_compile=False), num_samples)
    mcmc.run(windows, targets)
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    return predictive


def posterior_predictive_forward_and_backward_impl(
        windows_f: torch.tensor,
        targets_f: torch.tensor,
        windows_b: torch.tensor,
        targets_b: torch.tensor,
        num_samples: int = 100) -> Tuple[Predictive, Predictive]:
    predictive_f = get_samples_from_posterior_predictive(windows_f, targets_f, num_samples)
    predictive_b = get_samples_from_posterior_predictive(windows_b, targets_b, num_samples)
    return predictive_f, predictive_b


def posterior_predictive_forward_and_backward(
        train_d: BayesTrainData,
        save_dir: str,
        num_samples: int = 100) -> Tuple[Predictive, Predictive]:
    save_train_data_to_drive(train_d, save_dir)
    predictive_f, predictive_b = posterior_predictive_forward_and_backward_impl(
        windows_f=train_d.windows_f,
        targets_f=train_d.targets_f,
        windows_b=train_d.windows_b,
        targets_b=train_d.targets_b,
        num_samples=num_samples)
    torch.save(predictive_f, os.path.join(save_dir, "predictive.forward.torch"))
    torch.save(predictive_b, os.path.join(save_dir, "predictive.backward.torch"))
    return predictive_f, predictive_b


def train_logistic():
    posterior_predictive_forward_and_backward(
        train_d=BayesTrainData(load_logistic_map_time_series(1500),
                               window_len=1, noise_std=0.),
        save_dir="20230724_preds/logistics12", num_samples=60)


def train_garch():
    posterior_predictive_forward_and_backward(
        train_d=BayesTrainData(load_garch_time_series(2000, coef_alpha=0.1, rng_seed=42),
                               window_len=3),
        save_dir="20230724_preds/garch01")

    posterior_predictive_forward_and_backward(
        train_d=BayesTrainData(load_garch_time_series(2000, coef_alpha=0.7, rng_seed=42),
                               window_len=3),
        save_dir="20230724_preds/garch02")


if __name__ == "__main__":
    test_model_output_dimensions()
    train_logistic()
    # train_garch()
