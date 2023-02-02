import math
import numpy as np
import torch

from typing import Generator, Tuple
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def harmonic_oscillator(amplitude: float = 1.,
                        delta_t: float = 0.01,
                        initial_phase: float = 0.) -> Generator[float, None, None]:
    t = initial_phase
    while True:
        yield amplitude * math.sin(t)
        t += delta_t


def pull_data(generator: Generator[NDArray | float, None, None],
              n_points: int = 10000) -> NDArray:
    return np.fromiter(generator, dtype=np.float64, count=n_points)


def chop_time_series_into_windows(data: NDArray,
                                  window_len: int = 40,
                                  target_len: int = 1) -> Tuple[NDArray, NDArray]:
    assert data.ndim <= 2, "Time series expected, each datapoint is either a number or a 1D array"

    windows = np.array([data[i:i+window_len]
                        for i in range(len(data) - window_len - target_len + 1)])

    targets = np.array([data[i:i+target_len]
                        for i in range(window_len, len(data) - target_len + 1)])

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
    osc = harmonic_oscillator()
    t: NDArray = pull_data(osc, n_points=5)
    windows, targets = chop_time_series_into_windows(t, window_len=3, target_len=1)
    print(t)
    print(windows)
    print(targets)

    d = TimeSeriesDataset(windows, targets)
    print(d)
