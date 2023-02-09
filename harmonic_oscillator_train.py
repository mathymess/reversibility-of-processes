import datagen
from models import MyModel

import torch
import torch.nn as nn

import numpy as np

window_len = 20

# time_series = datagen.pull_data(datagen.harmonic_oscillator(), n_points=1000)
time_series = np.array(range(1000)) / 1000 * np.pi + 0.2 * np.random.rand(1000)

d = datagen.TimeSeriesDataset(*datagen.chop_time_series_into_windows(time_series,
                                                                     window_len=window_len))

num_epochs = 1000
loss_fn = nn.MSELoss()

model = MyModel(window_len=window_len, datapoint_size=1)
# model = nn.Sequential(nn.Linear(window_len, 100), nn.ReLU(), nn.Linear(100, 1))

dataloader = torch.utils.data.DataLoader(d, batch_size=20, shuffle=True)
# optim = torch.optim.SGD(model.parameters(), lr=0.1)
optim = torch.optim.Adam(model.parameters())


def test(model: MyModel, dataloader: torch.utils.data.DataLoader) -> float:
    losses = np.zeros(len(dataloader))
    with torch.no_grad():
        for i, (windows, targets) in enumerate(dataloader):
            pred = model(windows)
            loss = loss_fn(pred, targets.squeeze(0).to(torch.float32))
            losses[i] = loss.item()
    return losses.mean()


print("!!!", test(model, dataloader))

for epoch in range(num_epochs):
    for i, (windows, targets) in enumerate(dataloader):
        optim.zero_grad()

        pred = model(windows)

        loss = loss_fn(pred, targets.squeeze(0).to(torch.float32))
        loss.backward()

        optim.step()

        if i % 100 == 0:
            print(f"iteration {i} epoch {epoch} loss {loss}")

print("!!!", test(model, dataloader))
