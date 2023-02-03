import datagen
from model import MyModel

import numpy as np

import torch
import torch.nn as nn

from typing import Callable

window_len = 7
lrz_ts = datagen.generate_time_series_for_system(datagen.lorenz_attractor_ode,
                                                 initial_conditions=np.array([0., 1., 1.05]),
                                                 delta_t=1e-2,
                                                 n_points=10000)

data = datagen.TimeSeriesDataset(
        *datagen.chop_time_series_into_windows(lrz_ts, window_len=window_len))

d_train, d_test = torch.utils.data.random_split(data, [0.5, 0.5])


reversed_data = datagen.TimeSeriesDataset(
        *datagen.chop_time_series_into_windows(np.flip(lrz_ts.copy()), window_len=window_len))

rev_train, rev_test = torch.utils.data.random_split(reversed_data, [0.5, 0.5])

d_train_dl = torch.utils.data.DataLoader(d_train, batch_size=20, shuffle=True)
d_test_dl = torch.utils.data.DataLoader(d_test, batch_size=20, shuffle=True)
rev_train_dl = torch.utils.data.DataLoader(rev_train, batch_size=20, shuffle=True)
rev_test_dl = torch.utils.data.DataLoader(rev_test, batch_size=20, shuffle=True)

model = MyModel(window_len=window_len, datapoint_size=3)

def test(model: MyModel,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: Callable[..., float] = nn.MSELoss()) -> float:
    losses = np.zeros(len(dataloader))
    with torch.no_grad():
        for i, (windows, targets) in enumerate(dataloader):
            pred = model(windows)
            loss = loss_fn(pred, targets.squeeze(1))
            losses[i] = loss.item()
    return losses.mean()


def train(model: MyModel, dataloader, num_epochs = 20) -> None:
    loss_fn = nn.MSELoss()

    # optim = torch.optim.SGD(model.parameters(), lr=0.1)
    optim = torch.optim.Adam(model.parameters())

    print("!!!", test(model, dataloader, loss_fn))

    for epoch in range(num_epochs):
        for i, (windows, targets) in enumerate(dataloader):
            optim.zero_grad()

            pred = model(windows)

            loss = loss_fn(pred, targets.squeeze(1))
            loss.backward()

            optim.step()

        print(f"epoch {epoch} loss {loss}")

    print("!!!", test(model, dataloader, loss_fn))


train(model, d_train_dl)
print("Trained withhout reverse, loss on the test dataset:",
      test(model, d_test_dl))

train(model, rev_train_dl)
print("Trained with reverse, loss on the test dataset:",
      test(model, rev_test_dl))
