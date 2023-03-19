from models import ThreeFullyConnectedLayers
from datasets import TimeSeriesDataset

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from typing import Callable, List
CallbackType = Callable[[float], None]


def get_mean_loss_on_test_dataset(model: ThreeFullyConnectedLayers,
                                  test_dataset: torch.utils.data.Dataset,
                                  loss_fn: Callable[..., float] = nn.MSELoss()) -> float:
    losses = np.zeros(len(test_dataset))
    with torch.no_grad():
        for i, (window, target) in enumerate(test_dataset):
            losses[i] = loss_fn(model(window), target)
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


def empty_callback(scalar_value) -> None:
    pass


def train_loop_adam(model: nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    test_dataset: TimeSeriesDataset,
                    num_epochs: int = 20,
                    epochly_callback: CallbackType = empty_callback) -> None:
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(num_epochs):
        for i, (windows, targets) in enumerate(dataloader):
            optim.zero_grad()

            preds = model(windows)
            loss = loss_fn(preds, targets)
            loss.backward()

            optim.step()

        mean_test_loss = get_mean_loss_on_test_dataset(model, test_dataset)
        epochly_callback(mean_test_loss)

    model.eval()


def train_loop_adam_with_scheduler(model: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   test_dataset: TimeSeriesDataset,
                                   num_epochs: int = 20,
                                   epochly_callback: CallbackType = empty_callback) -> None:
    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

    model.train()
    for epoch in range(num_epochs):
        for i, (windows, targets) in enumerate(dataloader):
            optim.zero_grad()

            preds = model(windows)
            loss = loss_fn(preds, targets)
            loss.backward()

            optim.step()

        mean_test_loss = get_mean_loss_on_test_dataset(model, test_dataset)
        epochly_callback(mean_test_loss)
        scheduler.step()

    model.eval()


def train_loop_rmsprop(model: nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       test_dataset: TimeSeriesDataset,
                       num_epochs: int = 20,
                       epochly_callback: CallbackType = empty_callback) -> None:
    loss_fn = nn.MSELoss()
    optim = torch.optim.RMSprop(model.parameters())

    model.train()
    for epoch in range(num_epochs):
        for i, (windows, targets) in enumerate(dataloader):
            optim.zero_grad()

            preds = model(windows)
            loss = loss_fn(preds, targets)
            loss.backward()

            optim.step()

        mean_test_loss = get_mean_loss_on_test_dataset(model, test_dataset)
        epochly_callback(mean_test_loss)

    model.eval()


train_loop = train_loop_adam
