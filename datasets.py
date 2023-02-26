import numpy as np
import torch

import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def chop_time_series_into_chunks(time_series: NDArray,
                                 window_len: int = 40,
                                 take_each_nth_window: int = 1) -> NDArray:
    assert time_series.ndim == 2, "Time series expected, each datapoint is a 1D array"
    assert len(time_series) >= window_len, f"window_len={window_len} is too large"

    windows = np.array([time_series[i:i+window_len]
                        for i in range(0, len(time_series) - window_len + 1, take_each_nth_window)])

    return windows


def split_chunks_into_windows_and_targets(chunks: NDArray,
                                          target_len: int = 1,
                                          reverse: bool = False) -> tuple[NDArray, NDArray]:
    assert chunks.ndim == 3, "Should be (n_chunks, chunk_len, datapoint_dim)"

    chunk_len: int = chunks.shape[1]
    assert 0 < target_len < chunk_len, f"target_len={target_len} is too large or non-positive"

    if reverse:
        chunks = np.flip(chunks, 1)

    window_len: int = chunk_len - target_len

    windows: NDArray = chunks[:, :window_len, :]
    targets: NDArray = chunks[:, window_len:, :]

    return windows, targets


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, windows: NDArray, targets: NDArray) -> None:
        assert len(windows) == len(targets) != 0

        super().__init__()

        self.windows = torch.from_numpy(windows).to(torch.float32)
        self.targets = torch.from_numpy(targets).to(torch.float32)
        self.n_points: int = len(windows)

    def __getitem__(self, index: int) -> tuple:
        return (self.windows[index], self.targets[index])

    def __len__(self) -> int:
        return self.n_points

    def datapoint_torch_size(self):
        return self.windows[0][0]


def test_chop_time_series_into_chunks() -> None:
    def compare(actual: NDArray, expected: NDArray) -> None:
        assert np.array_equal(actual, expected), f"{actual} \n\t!=\n{expected}"

    simple_data = np.array([[1], [2], [3], [4]])
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

    compare(
        chop_time_series_into_chunks(simple_data, window_len=1, take_each_nth_window=1),
        np.array([[[1]], [[2]], [[3]], [[4]]])
    )

    compare(
        chop_time_series_into_chunks(simple_data, window_len=1, take_each_nth_window=3),
        np.array([[[1]], [[4]]])
    )

    compare(
        chop_time_series_into_chunks(simple_data, window_len=2, take_each_nth_window=1),
        np.array([[[1], [2]], [[2], [3]], [[3], [4]]])
    )

    compare(
        chop_time_series_into_chunks(simple_data, window_len=3, take_each_nth_window=1),
        np.array([[[1], [2], [3]], [[2], [3], [4]]])
    )

    compare(
        chop_time_series_into_chunks(data, window_len=1, take_each_nth_window=1),
        np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]], [[13, 14, 15]]])
    )

    compare(
        chop_time_series_into_chunks(data, window_len=2, take_each_nth_window=2),
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    )

    compare(
        chop_time_series_into_chunks(data, window_len=2, take_each_nth_window=3),
        np.array([[[1, 2, 3], [4, 5, 6]], [[10, 11, 12], [13, 14, 15]]])
    )


def test_split_chunks_into_windows_and_targets() -> None:
    def compare(actual: tuple[NDArray, NDArray],
                expected: tuple[NDArray, NDArray]) -> None:
        assert len(actual) == len(expected) == 2
        assert np.array_equal(actual[0], expected[0]), f"{actual} \n\t!=\n{expected}"
        assert np.array_equal(actual[1], expected[1]), f"{actual} \n\t!=\n{expected}"

    simple_data = np.array([[[1], [2], [3], [4]],
                            [[5], [6], [7], [8]],
                            [[9], [10], [11], [12]]])

    compare(
        split_chunks_into_windows_and_targets(simple_data, target_len=1),
        (
            np.array([[[1], [2], [3]],
                      [[5], [6], [7]],
                      [[9], [10], [11]]]),
            np.array([[[4]],
                      [[8]],
                      [[12]]])
        )
    )

    compare(
        split_chunks_into_windows_and_targets(simple_data, target_len=2),
        (
            np.array([[[1], [2]],
                      [[5], [6]],
                      [[9], [10]]]),
            np.array([[[3], [4]],
                      [[7], [8]],
                      [[11], [12]]])
        )
    )

    compare(
        split_chunks_into_windows_and_targets(simple_data, target_len=2, reverse=True),
        (
            np.array([[[4], [3]],
                      [[8], [7]],
                      [[12], [11]]]),
            np.array([[[2], [1]],
                      [[6], [5]],
                      [[10], [9]]])
        )
    )


if __name__ == "__main__":
    test_chop_time_series_into_chunks()
    test_split_chunks_into_windows_and_targets()
