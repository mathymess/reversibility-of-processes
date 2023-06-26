from models import ThreeFullyConnectedLayers
from datasets import (TimeSeriesDataset,
                      DataHolderOneDirection,
                      prepare_time_series_for_learning)

import lqrt
# https://github.com/alyakin314/lqrt/

import os
from pathlib import Path
import functools
import warnings
import json

import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from typing import Callable, List, Optional, Dict, Tuple, Iterable
import numpy.typing
CallbackType = Callable[[float], None]
NDArray = numpy.typing.NDArray[np.floating]


def get_mean_loss_on_test_dataset(model: ThreeFullyConnectedLayers,
                                  test_dataset: torch.utils.data.Dataset,
                                  loss_fn: Callable[..., float] = nn.MSELoss()) -> float:
    model.eval()
    losses = np.zeros(len(test_dataset))
    with torch.no_grad():
        for i, (window, target) in enumerate(test_dataset):
            losses[i] = loss_fn(model(window), target)

        model.train()
        return losses.mean()


class EpochlyCallback():
    def __init__(self, tensorboard_log_dir: str = "",
                 tensorboard_scalar_name: str = "mean_loss_on_train"):
        self.writer = tb.SummaryWriter(log_dir=tensorboard_log_dir)
        self.scalar_name = tensorboard_scalar_name
        self.all_values: List[float] = []

    def __call__(self, scalar_value: float) -> None:
        self.writer.add_scalar(self.scalar_name, scalar_value, len(self.all_values))
        self.all_values.append(scalar_value)
        self.writer.close()

    def get_values(self) -> List[float]:
        return self.all_values


class EpochlyCallbackBare():
    def __init__(self):
        self.all_values: List[float] = []

    def __call__(self, scalar_value: float) -> None:
        self.all_values.append(scalar_value)

    def get_values(self) -> List[float]:
        return self.all_values


def train_loop_adam_with_scheduler(model: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   test_dataset: TimeSeriesDataset,
                                   num_epochs: int = 20,
                                   epochly_callback: Optional[CallbackType] = None) -> None:
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.96)

    def epochly_callback_wrapped(m: nn.Module, d: TimeSeriesDataset) -> None:
        if epochly_callback is not None:
            model.eval()
            epochly_callback(get_mean_loss_on_test_dataset(m, d))
            model.train()

    epochly_callback_wrapped(model, test_dataset)
    for epoch in range(num_epochs):
        for i, (windows, targets) in enumerate(dataloader):
            optim.zero_grad()

            preds = model(windows)
            loss = loss_fn(preds, targets)
            loss.backward()

            optim.step()

        epochly_callback_wrapped(model, test_dataset)
        scheduler.step()

    model.eval()


def write_json_to_file(result: Dict, filepath: str) -> None:
    if filepath == "":
        warnings.warn(f"Not saving the following json because the filepath was empty: {result}")
        return

    dirname = os.path.dirname(filepath)
    try:
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)

        with open(filepath, "a") as file:
            json.dump(result, file)
    except IOError as e:
        warnings.warn(f"Tried to write to '{filepath}' but could not. Error: {e}")


def train_test_distribution(time_series: NDArray,
                            window_len: int,
                            target_len: int = 1,
                            hidden_size: int = 13,
                            num_epochs: int = 50,
                            num_runs: int = 100,
                            save_output_to_file: str = "") -> Dict:
    dh = prepare_time_series_for_learning(train_ts=time_series,
                                          test_ts=time_series.copy(),
                                          window_len=window_len,
                                          target_len=target_len,
                                          take_each_nth_chunk=1)
    train_loop = functools.partial(train_loop_adam_with_scheduler, num_epochs=num_epochs)

    def get_model() -> ThreeFullyConnectedLayers:
        return ThreeFullyConnectedLayers(window_len=window_len,
                                         target_len=target_len,
                                         datapoint_size=time_series.shape[-1],
                                         hidden_layer1_size=hidden_size,
                                         hidden_layer2_size=hidden_size)

    def get_forward_losses():
        m = get_model()
        callback = EpochlyCallbackBare()
        train_loop(m, dh.forward.train_loader, dh.forward.test_dataset, epochly_callback=callback)
        return callback.get_values()

    def get_backward_losses():
        m = get_model()
        callback = EpochlyCallbackBare()
        train_loop(m, dh.backward.train_loader, dh.backward.test_dataset, epochly_callback=callback)
        return callback.get_values()

    forward_losses, backward_losses = [], []
    for _ in range(num_runs):
        forward_losses.append(get_forward_losses())
        backward_losses.append(get_backward_losses())

    result = {"forward": forward_losses, "backward": backward_losses}
    write_json_to_file(result, save_output_to_file)

    return result


def train_test_distribution_montecarlo_ts(time_series_collection: Iterable[NDArray],
                                          window_len: int,
                                          target_len: int = 1,
                                          hidden_size: int = 13,
                                          datapoint_size: int = 3,
                                          num_epochs: int = 50,
                                          save_output_to_file: str = "") -> Dict:
    train_loop = functools.partial(train_loop_adam_with_scheduler, num_epochs=num_epochs)

    def get_model() -> ThreeFullyConnectedLayers:
        return ThreeFullyConnectedLayers(window_len=window_len,
                                         target_len=target_len,
                                         datapoint_size=datapoint_size,
                                         hidden_layer1_size=hidden_size,
                                         hidden_layer2_size=hidden_size)

    def get_losses(dho: DataHolderOneDirection):
        m = get_model()
        callback = EpochlyCallbackBare()
        train_loop(m, dho.train_loader, dho.test_dataset, epochly_callback=callback)
        return callback.get_values()

    forward_losses, backward_losses = [], []
    for time_series in time_series_collection:
        dh = prepare_time_series_for_learning(train_ts=time_series,
                                              test_ts=time_series.copy(),
                                              window_len=window_len,
                                              target_len=target_len,
                                              take_each_nth_chunk=1)
        forward_losses.append(get_losses(dh.forward))
        backward_losses.append(get_losses(dh.backward))

    result = {"forward": forward_losses, "backward": backward_losses}
    write_json_to_file(result, save_output_to_file)

    return result


class LossDistribution():
    def __init__(self, filepath: str) -> None:
        with open(filepath) as file:
            results = json.load(file)

        self.filepath = filepath
        self.label = Path(filepath).stem

        self.forward = np.array(results["forward"])
        self.backward = np.array(results["backward"])

        self.num_runs = self.forward.shape[0]
        self.num_epochs = self.forward.shape[1]

        assert self.forward.ndim == self.backward.ndim == 2
        assert self.forward.shape == self.backward.shape

    def plot_learning_curves(self, run_indices: List[int] = []) -> None:
        if not run_indices:
            run_indices = list(range(self.num_runs))

        for losses in self.forward[run_indices]:
            plt.plot(losses, label="forward", color="blue", alpha=0.5)
        for losses in self.backward[run_indices]:
            plt.plot(losses, label="backward", color="orange", alpha=0.5)

        plt.legend(handles=[mpatches.Patch(color="blue", label="forward"),
                            mpatches.Patch(color="orange", label="backward")])

        plt.grid()
        plt.title(self.label)
        plt.xlabel("epoch_number")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.show()

    def at_epoch(self, epoch: int) -> Tuple[NDArray, NDArray]:
        """Return array of losses at the same epoch, but different runs"""
        return self.forward[:, epoch], self.backward[:, epoch]

    def plot_distribution_at_epoch(self, epoch: int) -> None:
        f, b = self.at_epoch(epoch)

        plt.hist(f, alpha=0.5, edgecolor="black", label="forward")
        plt.hist(b, alpha=0.5, edgecolor="black", label="backward")

        plt.title(self.label + f" at epoch {epoch}")
        plt.xlabel("loss")
        plt.grid()
        plt.legend()
        plt.show()

    def wasserstein(self, epoch: int) -> float:
        return wasserstein_distance(self.forward[:, epoch], self.backward[:, epoch])

    def wasserstein_all(self) -> List[float]:
        return [self.wasserstein(epoch) for epoch in range(self.num_epochs)]

    def plot_wasserstein_vs_epoch(self, include_zeroth_epoch: bool = False) -> None:
        wasserstein_array = self.wasserstein_all()

        plt.plot(wasserstein_array, "o-")
        plt.grid()
        plt.title(self.label)
        plt.xlabel("epoch_number")
        plt.ylabel("wasserstein_distance")

        if not include_zeroth_epoch:
            plt.ylim(ymin=min(wasserstein_array) * 0.9, ymax=max(wasserstein_array[1:]) * 1.2)

        plt.show()

    def normalized_wasserstein(self, epoch: int) -> float:
        f, b = self.at_epoch(epoch)
        norm = np.mean(np.stack((f, b)))
        return wasserstein_distance(f, b) / norm

    def normalized_wasserstein_all(self) -> List[float]:
        return [self.normalized_wasserstein(epoch) for epoch in range(self.num_epochs)]

    def plot_normalized_wasserstein_vs_epoch(self) -> None:
        wasserstein_array = self.normalized_wasserstein_all()

        plt.plot(wasserstein_array, "o-")
        plt.grid()
        plt.title(self.label)
        plt.xlabel("epoch_number")
        plt.ylabel("normalized_wasserstein_distance")

    def relmeandiff_at_epoch(self, epoch: int) -> float:
        f, b = self.at_epoch(epoch)
        f_mean, b_mean = f.mean(), b.mean()
        return (b_mean - f_mean) / ((f_mean + b_mean) / 2)

    def relmeandiff_all(self) -> List[float]:
        return [self.relmeandiff_at_epoch(e) for e in range(self.num_epochs)]

    def plot_relmeandiff_vs_epoch(self) -> None:
        plt.plot(self.relmeandiff_all(), "o-")
        plt.grid()
        plt.title(self.label)
        plt.xlabel("epoch_number")
        plt.ylabel("relative difference in mean loss")
        plt.show()

    def lqrtest(self, epoch: int) -> lqrt.Lqrtest_indResult:  # type: ignore
        return lqrt.lqrtest_ind(self.forward[:, epoch],  # type: ignore
                                self.backward[:, epoch], equal_var=False)

    def lqrtest_all(self) -> List[lqrt.Lqrtest_indResult]:  # type: ignore
        return [self.lqrtest(epoch).pvalue for epoch in range(self.num_epochs)]

    def plot_lqrtest_pvalue_vs_epoch(self) -> None:
        lqr = self.lqrtest_all()
        plt.plot(lqr, "o-")
        plt.grid()
        plt.title(self.label)
        plt.xlabel("epoch_number")
        plt.ylabel("lqrt pvalue")
        plt.show()
