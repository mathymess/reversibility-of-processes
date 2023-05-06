from models import ThreeFullyConnectedLayers
from datasets import TimeSeriesDataset, prepare_time_series_for_learning

import functools
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from typing import Callable, List, Optional, Tuple
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


def train_loop_adam_with_scheduler(model: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   test_dataset: TimeSeriesDataset,
                                   num_epochs: int = 20,
                                   epochly_callback: Optional[CallbackType] = None) -> None:
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.96)

    def epochly_callback_wrapped() -> None:
        if epochly_callback is not None:
            model.eval()
            epochly_callback(get_mean_loss_on_test_dataset(model, test_dataset))
            model.train()

    epochly_callback_wrapped()
    for epoch in range(num_epochs):
        for i, (windows, targets) in enumerate(dataloader):
            optim.zero_grad()

            preds = model(windows)
            loss = loss_fn(preds, targets)
            loss.backward()

            optim.step()

        epochly_callback_wrapped()
        scheduler.step()

    model.eval()


def train_test_distribution(time_series: NDArray,
                            window_len: int,
                            target_len: int,
                            hidden_size: int,
                            train_loop: Callable = train_loop_adam_with_scheduler,
                            num_epochs: int = 10,
                            num_runs: int = 100,
                            save_output_to_file: str = "") -> Tuple[List[float], List[float]]:
    dh = prepare_time_series_for_learning(train_ts=time_series,
                                          test_ts=time_series.copy(),
                                          window_len=window_len,
                                          target_len=target_len,
                                          take_each_nth_chunk=1)
    train_loop = functools.partial(train_loop, num_epochs=num_epochs, epochly_callback=None)

    def get_model() -> ThreeFullyConnectedLayers:
        return ThreeFullyConnectedLayers(window_len=window_len,
                                         target_len=target_len,
                                         datapoint_size=time_series.shape[-1],
                                         hidden_layer1_size=hidden_size,
                                         hidden_layer2_size=hidden_size)

    def get_forward_loss():
        forward_model = get_model()
        train_loop(forward_model, dh.forward.train_loader, dh.forward.test_dataset)
        return get_mean_loss_on_test_dataset(forward_model, dh.forward.test_dataset)

    def get_backward_loss():
        backward_model = get_model()
        train_loop(backward_model, dh.backward.train_loader, dh.backward.test_dataset)
        return get_mean_loss_on_test_dataset(backward_model, dh.backward.test_dataset)

    forward_losses, backward_losses = [], []
    for i in range(num_runs):
        forward_losses.append(get_forward_loss())
        backward_losses.append(get_backward_loss())

    if save_output_to_file != "":
        try:
            with open(save_output_to_file, "a") as file:
                file.write(str((forward_losses, backward_losses)))
        except IOError as e:
            warnings.warn(f"Tried to write to '{save_output_to_file}' but could not. Error: {e}")

    return forward_losses, backward_losses
