import functools
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from typing import Callable, Optional, Tuple
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def harmonic_oscillator_ode(x: NDArray, _t: float, coef: float, friction: float) -> NDArray:
    assert x.shape == (2,)
    omega_sq = coef
    x_dot = np.zeros(2)  # x_dot[0] is the speed, x_dot[1] is acceleration
    x_dot[0] = x[1]
    x_dot[1] = - omega_sq * x[0] - friction * x[1]
    return x_dot


def belousov_zhabotinsky_ode(x: NDArray, _t: float,
                             coefs: Tuple[float, ...] = (5e-3, 0.6, 1 / 2e-2, 1 / 4e-4)) -> NDArray:
    assert x.shape == (3,)
    A, B, C, D = coefs
    x_dot = np.zeros(3)
    x_dot[0] = C * (x[1] * (A - x[0]) + x[0] * (1 - x[0]))
    x_dot[1] = D * (- x[1] * (A + x[0]) + B * x[2])
    x_dot[2] = x[0] - x[2]
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
                         coefs: Tuple[float, ...] = (10, 28, 2.667)) -> NDArray:
    # From https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html
    assert x.shape == (3,)
    A, B, C = coefs
    x_dot = np.zeros(3)
    x_dot[0] = A * (x[1] - x[0])
    x_dot[1] = B * x[0] - x[1] - x[0] * x[2]
    x_dot[2] = x[0] * x[1] - C * x[2]
    return x_dot


def double_pendulum_ode(x: NDArray, _t: float,
                        coefs: Tuple[float, ...],
                        friction: float) -> NDArray:
    # Code adopted from https://scipython.com/blog/the-double-pendulum/
    g = -9.81  # The gravitational acceleration, in $ms^{-2}$.
    L1, L2, m1, m2 = coefs  # Pendulum lengths (in m) and masses (in kg).

    # A double pendulum is described by generalized coordinates $\theta_1$ and $\theta_2$.
    # Denote their derivatives by $z_1$ and $z_2$ to reduce the ODE to the first order.
    theta1, theta2, z1, z2 = x

    c = np.cos(theta1 - theta2)
    s = np.sin(theta1 - theta2)

    theta1dot = z1
    theta2dot = z2

    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) - (m1+m2) *
             g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) +
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2) - theta2dot * friction / m1

    return np.array([theta1dot, theta2dot, z1dot, z2dot])


def generate_time_series_for_system(system: Callable[[NDArray, float], NDArray],
                                    initial_conditions: NDArray,
                                    t_shift: float = 0,
                                    t_density: float = 100,
                                    t_duration: float = 10000,
                                    second_order_ode_drop_half: bool = False,
                                    **kwargs) -> NDArray:
    t_linspace = np.linspace(0, t_duration, int(t_density * t_duration)) + t_shift
    sol = odeint(system, initial_conditions, t_linspace, **kwargs)

    if second_order_ode_drop_half:
        assert sol.shape[1] % 2 == 0
        halfdim = int(sol.shape[1] / 2)
        return sol[:, :halfdim]  # the second half is the derivatives

    return sol


def load_harmonic_oscillator_time_series(coef: float = 1.,
                                         initial_conditions: NDArray = np.array([0., 5.]),
                                         t_density: float = 5,
                                         t_duration: float = 200,
                                         friction: float = 0) -> NDArray:
    hos = generate_time_series_for_system(functools.partial(harmonic_oscillator_ode,
                                                            coef=coef,
                                                            friction=friction),
                                          initial_conditions=initial_conditions,
                                          t_density=t_density,
                                          t_duration=t_duration,
                                          second_order_ode_drop_half=True)
    return hos


def load_two_body_problem_time_series(coef: float = -1.,
                                      initial_conditions: NDArray = np.array([-1, 2, 0.2, 0.1]),
                                      t_density: float = 400,
                                      t_duration: float = 8) -> NDArray:
    twb = generate_time_series_for_system(functools.partial(two_body_problem_ode, coef=coef),
                                          initial_conditions=initial_conditions,
                                          t_density=t_density,
                                          t_duration=t_duration,
                                          second_order_ode_drop_half=True)
    return twb


def load_lorenz_attractor_time_series(coefs: Tuple[float, float, float] = (10, 28, 2.667),
                                      initial_conditions: NDArray = np.array([-10., -10, 28]),
                                      t_density: float = 100.,
                                      t_duration: float = 100.) -> NDArray:
    lrz = generate_time_series_for_system(functools.partial(lorenz_attractor_ode, coefs=coefs),
                                          initial_conditions=initial_conditions,
                                          t_density=t_density,
                                          t_duration=t_duration)
    return lrz


def load_belousov_zhabotinsky_time_series(
        coefs: Tuple[float, float, float, float] = (5e-3, 0.6, 1 / 2e-2, 1 / 4e-4),
        initial_conditions: NDArray = np.array([0., 17, 0.28]),
        t_density: float = 900.,
        t_duration: float = 4.) -> NDArray:
    bzh = generate_time_series_for_system(
        functools.partial(belousov_zhabotinsky_ode, coefs=coefs),
        initial_conditions=initial_conditions,
        t_density=t_density,
        t_duration=t_duration)

    return bzh


def load_double_pendulum_time_series(coefs: Tuple[float, float, float, float] = (1., 1., 1., 1.),
                                     friction: float = 0,
                                     initial_conditions: NDArray = np.array(
                                         [3*np.pi/7, 3*np.pi/4, 1, 5]),
                                     t_density: float = 40,
                                     t_duration: float = 250) -> NDArray:
    system = functools.partial(double_pendulum_ode, coefs=coefs, friction=friction)
    dbp = generate_time_series_for_system(system,
                                          initial_conditions=initial_conditions,
                                          t_density=t_density,
                                          t_duration=t_duration,
                                          second_order_ode_drop_half=True)
    # Normalize angles
    dbp %= 2 * np.pi
    dbp -= np.pi

    return dbp


def plot_3d_data(*data: NDArray,
                 title: str = "",
                 close_before_plotting: bool = True,
                 rainbow: bool = False,
                 show: bool = False) -> None:
    assert len(data) > 0, "Forgot to pass the np.array parameter"
    for d in data:
        assert d.ndim == 2 and d.shape[1] == 3, f"Expecting series of 3D data, got '{d}'"

    if close_before_plotting:
        plt.close()

    ax = plt.figure().add_subplot(projection="3d")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if title:
        plt.title(title)

    for d in data:
        ax.plot(*d.T, lw=0.5)
        if rainbow:
            ax.scatter(*d.T, lw=0.5, cmap="rainbow", c=range(d.shape[0]))
        else:
            ax.scatter(*d.T, lw=0.5)

    if show:
        plt.show()


def plot_2d_data(*data: NDArray,
                 title: str = "",
                 close_before_plotting: bool = True,
                 rainbow: bool = False,
                 show: bool = False) -> None:
    assert len(data) > 0, "Forgot to pass the np.array parameter"
    for d in data:
        assert d.ndim == 2 and d.shape[1] == 2, f"Expecting series of 2D data, got '{d}'"

    if close_before_plotting:
        plt.close()

    for d in data:
        plt.plot(*d.T, lw=0.5)
        if rainbow:
            plt.scatter(*d.T, lw=0.5, cmap="rainbow", c=range(d.shape[0]))
        else:
            plt.scatter(*d.T, lw=0.5)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

    if show:
        plt.show()


def plot_data_componentwise(*data: NDArray,
                            title: str = "Plotting multidimensional data componentwise",
                            close_before_plotting: bool = True,
                            show: bool = True,
                            draw_window_len: Optional[int] = None) -> None:
    assert len(data) > 0, "Forgot to pass the np.array parameter"
    n_components = data[0].shape[1]
    for d in data:
        assert d.ndim == 2, f"Expecting multidimensional data, got '{d}'"
        assert d.shape[1] == n_components, f"mismatched shapes: '{data}'"

    if close_before_plotting:
        plt.close()

    fig, ax = plt.subplots(n_components, squeeze=False)
    ax = ax.squeeze(axis=1)
    ax[0].set_title(title)

    for i in range(n_components):
        for d in data:
            ax[i].plot(d[:, i], "o-")

        ax[i].set_ylabel(f"$x_{i}$")
        ax[i].grid()

        if draw_window_len is not None:
            fractions = (0.3, 0.5, 0.8)
            x_axis_limit = max(len(d) for d in data)
            y_axis_limit = max(np.max(d[:, i]) for d in data)

            y = [y_axis_limit * f for f in fractions]
            x_min = [x_axis_limit * f for f in fractions]
            x_max = [x + draw_window_len for x in x_min]
            ax[i].hlines(y, x_min, x_max, color="red", linewidth=2)

    if show:
        plt.show()


def explore_harmonic_oscillator_time_series() -> None:
    hos = load_harmonic_oscillator_time_series(friction=0.02, t_duration=300)
    print(hos.shape)
    plot_data_componentwise(hos, title="Damped harmonic oscillator", show=True)


def explore_two_body_time_series() -> None:
    twb = load_two_body_problem_time_series()
    print(twb.shape)
    plot_2d_data(twb, title="2-body problem", show=True)
    plot_data_componentwise(twb, title="2-body problem, componentwise", show=True)


def explore_lorenz_attractor_time_series() -> None:
    lrz = load_lorenz_attractor_time_series()
    print(lrz.shape)
    plot_3d_data(lrz, title="Lorenz attractor time series", show=True,
                 rainbow=True)
    plot_data_componentwise(lrz, title="Lorenz attractor time series, componentwise", show=True)


def explore_belousov_zhabotinsky_time_series() -> None:
    bzh = load_belousov_zhabotinsky_time_series()
    print(bzh.shape)
    plot_3d_data(bzh, title="Belousov-Zhabotinsky time series",
                 show=True, rainbow=True)
    plot_data_componentwise(
        bzh, title="Belousov-Zhabotinsky time series, componentwise",
        show=True, draw_window_len=200)


def explore_double_pendulum_time_series() -> None:
    dbp = load_double_pendulum_time_series(friction=0.03)
    print(dbp.shape)
    n = 10000
    # plot_2d_data(dbp[:n], title=f"Double pendulum time series, first {n}", show=True)
    plot_data_componentwise(
        dbp[:n], title=f"Double pendulum time series, first {n}, componentwise",
        show=True, draw_window_len=200)


def load_logistic_map_time_series(length: int, coef_a: float = 4.0, x_initial: float = 0.6):
    x = np.zeros(length)
    x[0] = x_initial
    for i in range(1, length):
        x[i] = coef_a * x[i-1] * (1 - x[i-1])
    return x.reshape((x.size, 1))


if __name__ == "__main__":
    explore_two_body_time_series()
    explore_lorenz_attractor_time_series()
    explore_belousov_zhabotinsky_time_series()
    explore_double_pendulum_time_series()
    explore_harmonic_oscillator_time_series()
    plot_data_componentwise(load_logistic_map_time_series(1000))
