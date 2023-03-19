from generate_time_series import load_two_body_problem_time_series
from generate_time_series import load_belousov_zhabotinsky_time_series
from generate_time_series import load_lorenz_attractor_time_series
from datasets import chop_time_series_into_chunks, split_chunks_into_windows_and_targets

# # This function only works with discrete inputs (handy for categorization/clusterization).
# # It is unusable for the float vectors we are dealing with here.
# from sklearn.metrics import mutual_info_score

# https://www.blog.trainindata.com/mutual-information-with-python/
from sklearn.feature_selection import mutual_info_regression

from typing import Tuple
import numpy as np
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def time_series_2_windows_and_targets(time_series: NDArray,
                                      window_len: int = 30,
                                      target_len: int = 1,
                                      reverse: bool = False) -> Tuple[NDArray, NDArray]:
    chunks = chop_time_series_into_chunks(time_series,
                                          chunk_len=window_len+target_len,
                                          reverse=reverse,
                                          take_each_nth_chunk=1)
    windows, targets = split_chunks_into_windows_and_targets(chunks, target_len=target_len)
    return windows, targets


def calculate_mutual_info_for_dataset(ts: NDArray, dim: int = 0) -> Tuple[NDArray, NDArray]:
    assert 0 <= dim < ts.shape[1]

    forward_windows, forward_targets = time_series_2_windows_and_targets(ts)
    backward_windows, backward_targets = time_series_2_windows_and_targets(ts, reverse=True)

    backward_windows = backward_windows[:, :, dim]
    forward_windows = forward_windows[:, :, dim]
    forward_targets = forward_targets[:, 0, dim]
    backward_targets = backward_targets[:, 0, dim]

    return (mutual_info_regression(forward_windows, forward_targets),
            mutual_info_regression(backward_windows, backward_targets))


def print_mutual_info(ts: NDArray, comment: str) -> None:
    forward, backward = calculate_mutual_info_for_dataset(ts)
    print(comment, "forward", forward)
    print(comment, "backward", backward)


if __name__ == "__main__":
    print_mutual_info(load_two_body_problem_time_series(), "kepler")
    print()
    print_mutual_info(load_belousov_zhabotinsky_time_series(), "belousov_zhabotinsky")
    print()
    print_mutual_info(load_lorenz_attractor_time_series(), "lorenz")

    # The numbers are roughly the same for both forward/backward directions.
