from models import ThreeFullyConnectedLayers
from datasets import TimeSeriesDataset, prepare_time_series_for_learning

import os
import functools
import warnings
import json
import numpy as np

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from typing import Callable, List, Optional, Dict
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
    for i in range(num_runs):
        forward_losses.append(get_forward_losses())
        backward_losses.append(get_backward_losses())

    result = {"forward": forward_losses, "backward": backward_losses}
    if save_output_to_file != "":
        try:
            os.makedirs(os.path.dirname(save_output_to_file), exist_ok=True)
            with open(save_output_to_file, "a") as file:
                json.dump(result, file)
        except IOError as e:
            warnings.warn(f"Tried to write to '{save_output_to_file}' but could not. Error: {e}")

    return result
