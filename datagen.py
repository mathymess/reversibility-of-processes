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


def pull_data(generator: Generator[float, None, None],
              n_points: int = 10000) -> NDArray:
    return np.fromiter(generator, dtype=float, count=n_points)


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: NDArray,
                 window_size: int = 40,
                 prediction_size: int = 1,
                 is_reversed: bool = False) -> None:
        super().__init__()

        self.data: NDArray = data
        self.window_size: int = window_size
        self.prediction_size: int = prediction_size
        self.is_reversed: bool = is_reversed

    def __len__(self) -> int:
        return len(self.data)

    def __get_item__(self, index: int) -> Tuple[NDArray, NDArray]:
        return (np.array([1.]), np.array([1.]))


if __name__ == "__main__":
    osc = harmonic_oscillator()
    lol: NDArray = pull_data(osc)
