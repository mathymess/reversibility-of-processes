from generate_time_series import *
from datasets import *
from models import MyModel

import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from typing import Callable


def load_lorenz_attractor_dataholder(chunk_len: int, shift_ratio: float) -> AllDataHolder:
    assert 0 < shift_ratio < 1
    lrz = load_lorenz_attractor_time_series()
    lrz_train, lrz_test = train_test_split(lrz, shift=int(len(lrz) * shift_ratio))
    dh = prepare_time_series_for_learning(lrz_train, lrz_test, chunk_len=chunk_len)
    return dh


def load_belousov_zhabotinsky_dataholder(chunk_len: int, shift_ratio: float = 0) -> AllDataHolder:
    assert 0 <= shift_ratio < 1
    bzh = load_belousov_zhabotinsky_time_series()
    bzh_train, bzh_test = train_test_split(bzh, shift=int(len(bzh) * shift_ratio))
    dh = prepare_time_series_for_learning(bzh_train, bzh_test, chunk_len=chunk_len)
    return dh


def load_two_body_problem_dataholder(chunk_len: int, shift_ratio: float = 0) -> AllDataHolder:
    assert 0 <= shift_ratio < 1
    twb = load_two_body_problem_time_series()
    twb_train, twb_test = train_test_split(twb, shift=int(len(twb) * shift_ratio))
    dh = prepare_time_series_for_learning(twb_train, twb_test, chunk_len=chunk_len)
    return dh


def get_mean_loss_on_test_dataset(model: MyModel,
                                  test_dataset: torch.utils.data.Dataset,
                                  loss_fn: Callable[..., float] = nn.MSELoss()) -> float:
    losses = np.zeros(len(test_dataset))
    with torch.no_grad():
        for i, (window, target) in enumerate(test_dataset):
            losses[i] = loss_fn(model(window), target.squeeze(dim=0))
        return losses.mean()


def train(model: MyModel,
          dataloader: torch.utils.data.DataLoader,
          num_epochs: int = 20,
          tensorboard_dir_suffix: str = "") -> None:
    tensorboard_dir = "runs/" + time.strftime("%Y%m%d_%H%M%S") + "." + tensorboard_dir_suffix
    writer = SummaryWriter(log_dir=tensorboard_dir)

    loss_fn = nn.MSELoss()

    # optim = torch.optim.SGD(model.parameters(), lr=0.1)
    optim = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for i, (windows, targets) in enumerate(dataloader):
            optim.zero_grad()

            preds = model(windows)
            loss = loss_fn(preds, targets.squeeze(dim=1))
            loss.backward()

            optim.step()

        mean_test_loss = get_mean_loss_on_test_dataset(model, dh.forward.test_dataset)
        writer.add_scalar("loss_on_test",  mean_test_loss, epoch)

    writer.close()


if __name__ == "__main__":
    window_len = 30
    dh = load_two_body_problem_dataholder(chunk_len=window_len+1,
                                          shift_ratio=0.9)
    # plot_3d_data(dh.train_ts, dh.test_ts, show=True, title="Lorenz time series, train/test")
    # plot_data_componentwise(dh.train_ts, dh.test_ts, draw_window_len=40,
    #                         show=True, title="Lorenz time series, train/test")

    train(MyModel(window_len=window_len, datapoint_size=2),
          dh.forward.train_loader,
          num_epochs=20,
          tensorboard_dir_suffix="kepler_forward")

    train(MyModel(window_len=window_len, datapoint_size=2),
          dh.backward.train_loader,
          num_epochs=20,
          tensorboard_dir_suffix="kepler_backward")

# model = MyModel(window_len=window_len, datapoint_size=3)
# train(model, d_train_dl, tensorboard_dir_suffix="direct")
# print("Trained without reverse, loss on the test dataset:",
#       test(model, d_test_dl))
#
# model = MyModel(window_len=window_len, datapoint_size=3)
# train(model, rev_train_dl, tensorboard_dir_suffix="reverse")
# print("Trained with reverse, loss on the test dataset:",
#       test(model, rev_test_dl))
