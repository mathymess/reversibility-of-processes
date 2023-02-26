import datagen

import numpy as np
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


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
            t_density=100,
            t_duration=10)

    print(d)
    datagen.plot_data_componentwise(d, show=True)
    datagen.plot_3d_data(d, show=True)
