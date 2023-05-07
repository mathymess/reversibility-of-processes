from generate_time_series import (load_two_body_problem_time_series,
                                  load_lorenz_attractor_time_series,
                                  load_belousov_zhabotinsky_time_series,
                                  load_double_pendulum_time_series)
from train_test_utils import train_test_distribution

import threading
import functools

from typing import Tuple
import numpy.typing
NDArray = numpy.typing.NDArray[numpy.floating]


def calculate_distributions(time_series: NDArray,
                            filename_prefix: str,
                            hidden_layer_size_mesh: Tuple[int, ...] = (13,),
                            window_len_mesh: Tuple[int, ...] = (5, 12, 25)) -> None:
    for window_len in window_len_mesh:
        for size in hidden_layer_size_mesh:
            filepath = (f"20230507_distributions/{filename_prefix}_"
                        f"size={size}_window_len={window_len}.json")
            task = functools.partial(train_test_distribution,
                                     time_series=time_series,
                                     window_len=window_len,
                                     save_output_to_file=filepath,
                                     num_epochs=30)
            threading.Thread(target=task).run()


def main() -> None:
    calculate_distributions(time_series=load_two_body_problem_time_series(),
                            filename_prefix="kepler")

    calculate_distributions(time_series=load_lorenz_attractor_time_series(),
                            filename_prefix="lorenz")

    calculate_distributions(time_series=load_belousov_zhabotinsky_time_series(),
                            filename_prefix="belousovzhabotinsky")

    calculate_distributions(time_series=load_double_pendulum_time_series(),
                            filename_prefix="doublependulum")


if __name__ == "__main__":
    main()
