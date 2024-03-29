import os
import numpy as np
import numpy.typing
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import Predictive
from pyro.contrib import forecast as pyro_metric

from generate_time_series import load_logistic_map_time_series, load_garch_time_series
from datasets import chop_time_series_into_chunks, split_chunks_into_windows_and_targets


NDArray = numpy.typing.NDArray[numpy.floating]


def prepare_simple_1d_time_series(ts: NDArray,
                                  window_len: int,
                                  take_each_nth_chunk: int = 1,
                                  reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, windows: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert windows.ndim in (1, 2)
        assert windows.shape[-1] == self.input_size

        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # Infer the response noise

        ret = windows
        ret = self.relu1(self.fc1(ret))
        ret = self.relu2(self.fc2(ret))
        ret = self.fc3(ret)

        with pyro.plate("data", windows.shape[0]):
            obs = pyro.sample("obs",  # noqa: F841
                              dist.Normal(ret.squeeze(-1), sigma * sigma),
                              obs=None if y is None else y.squeeze(-1))

        return ret


def test_model_output_dimensions() -> None:
    model = BayesianThreeFCLayers(window_len=2, datapoint_size=1, target_len=1)
    batch = torch.tensor([[1., 2.], [3., 4.]])
    targets = model(batch)
    assert targets.ndim == 2 and targets.shape == (2, 1), f"targets.shape={targets.shape}"
    print("test_model_output_dimensions passed successfully!")


def get_samples_from_posterior_predictive(windows: torch.Tensor,
                                          targets: torch.Tensor,
                                          num_samples: int = 100,
                                          hidden_size: int = 10) -> Predictive:
    assert windows.ndim == targets.ndim == 2
    assert windows.shape[0] == targets.shape[0]

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    model = BayesianThreeFCLayers(window_len=windows.shape[-1], target_len=1,
                                  datapoint_size=1, hidden_size=hidden_size)
    mcmc = pyro.infer.MCMC(pyro.infer.NUTS(model, jit_compile=False), num_samples)
    mcmc.run(windows, targets)
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    return predictive


def posterior_predictive_forward_and_backward_impl(
        windows_f: torch.Tensor,
        targets_f: torch.Tensor,
        windows_b: torch.Tensor,
        targets_b: torch.Tensor,
        num_samples: int = 100,
        hidden_size: int = 10) -> Tuple[Predictive, Predictive]:
    predictive_f = get_samples_from_posterior_predictive(windows_f, targets_f,
                                                         num_samples=num_samples,
                                                         hidden_size=hidden_size)
    predictive_b = get_samples_from_posterior_predictive(windows_b, targets_b,
                                                         num_samples=num_samples,
                                                         hidden_size=hidden_size)
    return predictive_f, predictive_b


def posterior_predictive_forward_and_backward(
        train_d: BayesTrainData,
        save_dir: str,
        num_samples: int = 100,
        hidden_size: int = 10) -> Tuple[Predictive, Predictive]:
    save_train_data_to_drive(train_d, save_dir)
    predictive_f, predictive_b = posterior_predictive_forward_and_backward_impl(
        windows_f=train_d.windows_f,
        targets_f=train_d.targets_f,
        windows_b=train_d.windows_b,
        targets_b=train_d.targets_b,
        num_samples=num_samples,
        hidden_size=hidden_size)
    torch.save(predictive_f, os.path.join(save_dir, "predictive.forward.torch"))
    torch.save(predictive_b, os.path.join(save_dir, "predictive.backward.torch"))
    return predictive_f, predictive_b


class ExperimentResults:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir

        self.windows_f = torch.load(os.path.join(save_dir, "windows_f.torch"),
                                    map_location=torch.device('cpu'))
        self.targets_f = torch.load(os.path.join(save_dir, "targets_f.torch"),
                                    map_location=torch.device('cpu'))
        self.predictive_f = torch.load(os.path.join(save_dir, "predictive.forward.torch"),
                                       map_location=torch.device('cpu'))
        self.pred_obs_f = self.predictive_f(self.windows_f)["obs"]

        self.windows_b = torch.load(os.path.join(save_dir, "windows_b.torch"),
                                    map_location=torch.device('cpu'))
        self.targets_b = torch.load(os.path.join(save_dir, "targets_b.torch"),
                                    map_location=torch.device('cpu'))
        self.predictive_b = torch.load(os.path.join(save_dir, "predictive.backward.torch"),
                                       map_location=torch.device('cpu'))
        self.pred_obs_b = self.predictive_b(self.windows_b)["obs"]

        try:
            self.ts = torch.load(os.path.join(save_dir, "ts.torch"),
                                 map_location=torch.device('cpu'))
            self.noisy_ts = torch.load(os.path.join(save_dir, "noisy_ts.torch"),
                                       map_location=torch.device('cpu'))
        except FileNotFoundError:
            pass


def plot_predictions(true: torch.Tensor,
                     pred_all: Optional[torch.Tensor] = None,
                     pred_mean: Optional[torch.Tensor] = None,
                     pred_std: Optional[torch.Tensor] = None,
                     show: bool = False,
                     xlim: Optional[Tuple[float, float]] = None,
                     ylim: Optional[Tuple[float, float]] = None,
                     title: Optional[str] = None) -> plt.Axes:
    fig, ax = plt.subplots()

    if pred_all is not None:
        for pred_onedraw in torch.stack(list(pred_all)):
            ax.plot(pred_onedraw, "o-", linewidth=1, markersize=1,
                    alpha=0.2, color="green")

    if pred_std is not None and pred_mean is not None:
        top = (pred_mean + pred_std).squeeze(-1)
        bottom = (pred_mean - pred_std).squeeze(-1)
        print(top.shape, bottom.shape)
        ax.fill_between(range(len(top)), bottom, top, alpha=0.6, color="#86cfac", zorder=5)

    if pred_mean is not None:
        ax.plot(pred_mean, 'ro-', linewidth=1, markersize=1, label="predictive mean")

    ax.plot(true, 'bo-', linewidth=1, markersize=1, label="true value")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel("index")
    ax.set_ylabel("target value")
    ax.grid()
    ax.legend()

    if title is not None:
        ax.set_title(title)
    if show:
        fig.show()

    return fig, ax


def quality_metrics(preds: torch.Tensor, truth: torch.Tensor) -> Dict:
    err_str = f"preds.shape={preds.shape}, truth.shape={truth.shape}"
    # assert preds.ndim == truth.ndim + 1 == 3, err_str
    assert preds.ndim == truth.ndim == 2, err_str
    assert preds.shape[1] == truth.shape[0], err_str
    truth = truth.squeeze(-1)

    metrics = {}
    metrics["mae"] = pyro_metric.eval_mae(preds, truth)
    metrics["rmse"] = pyro_metric.eval_rmse(preds, truth)
    metrics["crps"] = pyro_metric.eval_crps(preds, truth)
    metrics["mean_std"] = preds.std(axis=0).mean().cpu().item()
    metrics["mse_from_all"] = ((truth - preds) ** 2).mean().cpu().item()

    return metrics


def train_logistic():
    posterior_predictive_forward_and_backward(
        train_d=BayesTrainData(load_logistic_map_time_series(1500),
                               window_len=1, noise_std=0.05),
        save_dir="20230724_preds/logistics14", num_samples=60)


def train_garch():
    posterior_predictive_forward_and_backward(
        train_d=BayesTrainData(load_garch_time_series(2000, coef_alpha=0.1, rng_seed=42),
                               window_len=3, noise_std=0.),
        save_dir="20230724_preds/garch03")

    # posterior_predictive_forward_and_backward(
    #     train_d=BayesTrainData(load_garch_time_series(2000, coef_alpha=0.7, rng_seed=42),
    #                            window_len=3),
    #     save_dir="20230724_preds/garch02")


if __name__ == "__main__":
    test_model_output_dimensions()
    # train_logistic()
    # train_garch()
