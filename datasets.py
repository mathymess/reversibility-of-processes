import datagen

import numpy as np
from matplotlib import pyplot as plt

import torch

from typing import Tuple
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def chop_time_series_into_windows(data: NDArray,
                                  window_len: int = 40,
                                  target_len: int = 1,
                                  skip_till_each_nth=None) -> Tuple[NDArray, NDArray]:
    assert data.ndim <= 2, "Time series expected, each datapoint is either a number or a 1D array"
    if skip_till_each_nth is None:
        skip_till_each_nth = int(window_len / 2)

    windows = np.array([data[i:i+window_len]
                        for i in range(len(data) - window_len - target_len + 1)])

    targets = np.array([data[i:i+target_len]
                        for i in range(window_len, len(data) - target_len + 1)])

    windows = windows[::skip_till_each_nth]
    targets = targets[::skip_till_each_nth]

    assert len(windows) == len(targets)
    return windows, targets


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, windows: NDArray, targets: NDArray) -> None:
        assert len(windows) == len(targets) != 0

        super().__init__()

        self.windows = torch.from_numpy(windows).to(torch.float32)
        self.targets = torch.from_numpy(targets).to(torch.float32)
        self.n_points: int = len(windows)

    def __getitem__(self, index: int) -> Tuple[NDArray, NDArray]:
        return (self.windows[index], self.targets[index])

    def __len__(self) -> int:
        return self.n_points

    def datapoint_torch_size(self):
        return self.windows[0][0]


if __name__ == "__main__":
    t: NDArray = pull_data_from_generator(harmonic_oscillator(), n_points=5)
    windows, targets = chop_time_series_into_windows(t, window_len=3, target_len=1)
    # print(t)
    # print(windows)
    # print(targets)

    d = TimeSeriesDataset(windows, targets)
    # print(d)
