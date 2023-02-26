import datagen

import numpy as np
from matplotlib import pyplot as plt

import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def plot_3d_data(data: NDArray) -> None:
    assert len(data) == 2 and data.shape[1] == 3, "Expecting 3-dimensional data"

    plt.close()
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(*data.T, lw=0.5)
    ax.scatter(*data.T, lw=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def plot_data_componentwise(data: NDArray,
                            title: str = "Plotting multidimensional data componentwise") -> None:
    assert data.ndim == 2, "Expecting multidimensional data"

    plt.close()
    n_components = data.shape[1]
    fig, ax = plt.subplots(n_components)
    ax[0].set_title(title)

    for i in range(n_components):
        ax[i].plot(data[:, i])
        ax[i].scatter(range(len(data)), data[:, i])
        ax[i].set_ylabel('xyzp'[i])
        ax[i].grid()


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


d = datagen.generate_time_series_for_system(
        belousov_zhabotinsky_simplified,
        initial_conditions=np.array([0.5, 0.5, 0.5]),
        delta_t=5e-3)


if __name__ == "__main__":
    print(d)
    plot_data_componentwise(d)
# plt.show()
