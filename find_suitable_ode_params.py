import datagen

import numpy as np
from matplotlib import pyplot as plt

import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def plot_3d_data(data: NDArray, close_before_plotting: bool = True, show: bool = False) -> None:
    assert data.ndim == 2 and data.shape[1] == 3, "Expecting 3-dimensional data"

    if close_before_plotting:
        plt.close()

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(*data.T, lw=0.5)
    ax.scatter(*data.T, lw=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if show:
        plt.show()


def plot_2d_data(data: NDArray, close_before_plotting: bool = True, show: bool = False) -> None:
    assert data.ndim == 2 and data.shape[1] == 2, "Expecting 2-dimensional data"

    if close_before_plotting:
        plt.close()

    plt.plot(*data.T, lw=0.5)
    plt.scatter(*data.T, lw=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid('y')

    if show:
        plt.show()


def plot_data_componentwise(data: NDArray,
                            title: str = "Plotting multidimensional data componentwise",
                            close_before_plotting: bool = True,
                            show: bool = True) -> None:
    assert data.ndim == 2, "Expecting multidimensional data"

    if close_before_plotting:
        plt.close()

    n_components = data.shape[1]
    fig, ax = plt.subplots(n_components)
    ax[0].set_title(title)

    for i in range(n_components):
        ax[i].plot(data[:, i])
        ax[i].scatter(range(len(data)), data[:, i])
        ax[i].set_ylabel("xyzp"[i])
        ax[i].grid()

    if show:
        plt.show()


def belousov_zhabotinsky_with_coefs(x: NDArray, _t: float) -> NDArray:
    coefs = (10., -3., 4., -0.879, -10.)
    return datagen.belousov_zhabotinsky_ode(x, _t, coefs=coefs)


def belousov_zhabotinsky_simplified(x: NDArray, _t: float) -> NDArray:
    # coefs = (8e-4, 0.666, 1 / 4e-2, 1 / 4e-4)
    coefs = (5e-3, 0.6, 1 / 2e-2, 1 / 4e-4)
    A, B, C, D = coefs
    x_dot = np.zeros(3)
    x_dot[0] = C * (x[1] * (A - x[0]) + x[0] * (1 - x[0]))
    x_dot[1] = D * (- x[1] * (A + x[0]) + B * x[2])
    x_dot[2] = x[0] - x[2]
    return x_dot


if __name__ == "__main__":
    d = datagen.generate_time_series_for_system(
            belousov_zhabotinsky_simplified,
            initial_conditions=np.array([0.5, 0.5, 0.5]),
            t_density=200)

    print(d)
    plot_data_componentwise(d)
    # plt.show()
