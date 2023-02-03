import math
import numpy as np
from scipy.integrate import odeint

import torch

from typing import Callable, Generator, Tuple
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def harmonic_oscillator(amplitude: float = 1.,
                        delta_t: float = 0.01,
                        initial_phase: float = 0.) -> Generator[float, None, None]:
    t = initial_phase
    while True:
        yield amplitude * math.sin(t)
        t += delta_t


def pull_data_from_generator(generator: Generator[NDArray | float, None, None],
                             n_points: int = 10000) -> NDArray:
    return np.fromiter(generator, dtype=np.float64, count=n_points)


def harmonic_oscillator_ode(x: NDArray, _t: float) -> NDArray:
    assert x.shape == (2,)
    omega_sq = 1
    x_deriv = np.zeros(2)
    x_deriv[0] = x[1]
    x_deriv[1] = - omega_sq * x[0]
    return x_deriv


def belousov_zhabotinsky_ode(x: NDArray, _t: float) -> NDArray:
    assert x.shape == (3,)
    A, B, C, D, E = -0.7, -0.5, 0.1, 0.3, 0.09
    x_deriv = np.zeros(3)
    x_deriv[0] = A * x[0] * (C - x[1]) - B * x[0] * x[2]
    x_deriv[1] = A * x[0] * (C - x[1]) - D * x[1]
    x_deriv[2] = D * x[1] - E * x[2]
    return x_deriv


def two_body_problem_ode(x: NDArray, _t: float) -> NDArray:
    # Simplified 2D ODE: $\ddot{\vec{r}} = \frac{C \vec{r}}{|\vec{r}|^3}
    # $\vec{r}$ = (x[0], x[1])
    # $\dot{\vec{r}}$ = (x[0], x[1])
    assert x.shape == (4,)
    C = 1
    dist = np.linalg.norm(x[:2])
    x_deriv = np.zeros(4)
    x_deriv[0] = x[2]
    x_deriv[1] = x[3]
    x_deriv[2] = x[0] * C / dist ** 3
    x_deriv[3] = x[1] * C / dist ** 3
    return x_deriv


def generate_time_series_for_system(system: Callable[[NDArray, float], NDArray],
                                    initial_conditions: NDArray,
                                    n_points: int = 10000,
                                    second_order_ode_drop_half: bool = False) -> NDArray:
    sol = odeint(system, initial_conditions, np.linspace(0, 500, n_points))

    if second_order_ode_drop_half:
        assert sol.shape[1] % 2 == 0
        halfdim = int(sol.shape[1] / 2)
        return sol[:, :halfdim]  # the second half is the derivatives

    return sol


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
    t: NDArray = pull_data_from_generator(osc, n_points=5)
    windows, targets = chop_time_series_into_windows(t, window_len=3, target_len=1)
    print(t)
    print(windows)
    print(targets)

    d = TimeSeriesDataset(windows, targets)
    print(d)
