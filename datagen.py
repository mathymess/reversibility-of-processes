import math
import numpy as np
from scipy.integrate import odeint

from typing import Callable, Generator
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
    x_dot = np.zeros(2)
    x_dot[0] = x[1]
    x_dot[1] = - omega_sq * x[0]
    return x_dot


def belousov_zhabotinsky_ode(x: NDArray, _t: float,
                             coefs: tuple[float, ...] = (10., -3., 4., -0.879, -10.)) -> NDArray:
    assert x.shape == (3,)
    A, B, C, D, E = coefs
    x_dot = np.zeros(3)
    x_dot[0] = A * x[0] * (C - x[1]) - B * x[0] * x[2]
    x_dot[1] = A * x[0] * (C - x[1]) - D * x[1]
    x_dot[2] = D * x[1] - E * x[2]
    return x_dot


def two_body_problem_ode(x: NDArray, _t: float, coef: float = -1) -> NDArray:
    r"""
    Simplified 2D ODE: $\ddot{\vec{r}} = \frac{C \vec{r}}{|\vec{r}|^3}
    $\vec{r}$ = (x[0], x[1])
    $\dot{\vec{r}}$ = (x[0], x[1])
    """
    assert x.shape == (4,)

    dist = np.linalg.norm(x[:2])
    x_dot = np.zeros(4)

    x_dot[0] = x[2]
    x_dot[1] = x[3]
    x_dot[2] = x[0] * coef / dist ** 3
    x_dot[3] = x[1] * coef / dist ** 3

    return x_dot


def lorenz_attractor_ode(x: NDArray, _t: float,
                         coefs: tuple[float, ...] = (10, 28, 2.667)) -> NDArray:
    # From https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html
    assert x.shape == (3,)
    A, B, C = coefs
    x_dot = np.zeros(3)
    x_dot[0] = A * (x[1] - x[0])
    x_dot[1] = B * x[0] - x[1] - x[0] * x[2]
    x_dot[2] = x[0] * x[1] - C * x[2]
    return x_dot


def generate_time_series_for_system(system: Callable[[NDArray, float], NDArray],
                                    initial_conditions: NDArray,
                                    t_shift: float = 0,
                                    t_density: float = 100,
                                    t_duration: int = 10000,
                                    second_order_ode_drop_half: bool = False,
                                    **kwargs) -> NDArray:
    t_linspace = np.linspace(0, t_duration, t_density * t_duration) + t_shift
    sol = odeint(system, initial_conditions, t_linspace, **kwargs)

    if second_order_ode_drop_half:
        assert sol.shape[1] % 2 == 0
        halfdim = int(sol.shape[1] / 2)
        return sol[:, :halfdim]  # the second half is the derivatives

    return sol


def load_two_body_problem_time_series() -> NDArray:
    twb = generate_time_series_for_system(two_body_problem_ode,
                                          initial_conditions=np.array([1, 3, -0.1, 0.1]),
                                          t_density=400,
                                          t_duration=13,
                                          second_order_ode_drop_half=True)
    return twb


if __name__ == "__main__":
    twb = load_two_body_problem_time_series()
