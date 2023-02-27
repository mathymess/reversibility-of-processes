import numpy as np
import torch

from dataclasses import dataclass
from typing import Optional
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def chop_time_series_into_chunks(time_series: NDArray,
                                 chunk_len: int,
                                 take_each_nth_chunk: int) -> NDArray:
    assert time_series.ndim == 2, "Time series expected, each datapoint is a 1D array"
    assert len(time_series) >= chunk_len, f"chunk_len={chunk_len} is too large"

    chunks = np.array([time_series[i:i+chunk_len]
                      for i in range(0, len(time_series) - chunk_len + 1, take_each_nth_chunk)])

    return chunks


def split_chunks_into_windows_and_targets(chunks: NDArray,
                                          target_len: int = 1,
                                          reverse: bool = False) -> tuple[NDArray, NDArray]:
    assert chunks.ndim == 3, "Shape should be (n_chunks, chunk_len, datapoint_dim)"

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
        assert windows.ndim == 3
        assert targets.ndim == 3
        assert windows.shape[2] == targets.shape[2]

        super().__init__()

        self.windows = torch.from_numpy(windows).to(torch.float32)
        self.targets = torch.from_numpy(targets).to(torch.float32)
        self.n_points: int = len(windows)

    def __getitem__(self, index: int) -> tuple:
        return (self.windows[index], self.targets[index])

    def __len__(self) -> int:
        return self.n_points


def time_series_to_dataset(ts: NDArray,
                           chunk_len: int,
                           take_each_nth_chunk: int,
                           reverse: bool = False) -> TimeSeriesDataset:
    chunks = chop_time_series_into_chunks(ts, chunk_len=chunk_len,
                                          take_each_nth_chunk=take_each_nth_chunk)
    windows, targets = split_chunks_into_windows_and_targets(chunks, reverse=reverse)
    return TimeSeriesDataset(windows, targets)


@dataclass
class DataHolderOneDirection:
    train_dataset: TimeSeriesDataset
    test_dataset: TimeSeriesDataset
    train_loader: torch.utils.data.DataLoader


@dataclass
class AllDataHolder:
    forward: DataHolderOneDirection
    backward: DataHolderOneDirection

    test_ts: Optional[NDArray] = None
    train_ts: Optional[NDArray] = None


def prepare_time_series_for_learning(train_ts: NDArray,
                                     test_ts: NDArray,
                                     chunk_len: int = 40,
                                     loader_batch_size: int = 20) -> AllDataHolder:
    assert chunk_len >= 2
    assert loader_batch_size >= 1

    take_each_nth_chunk: int = chunk_len // 2

    # Forward
    train_dataset = time_series_to_dataset(train_ts, chunk_len=chunk_len,
                                           take_each_nth_chunk=take_each_nth_chunk)
    test_dataset = time_series_to_dataset(test_ts, chunk_len=chunk_len,
                                          take_each_nth_chunk=take_each_nth_chunk)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=loader_batch_size)
    forward = DataHolderOneDirection(train_dataset, test_dataset, train_loader)

    # Backward
    train_dataset = time_series_to_dataset(train_ts, chunk_len=chunk_len, reverse=True,
                                           take_each_nth_chunk=take_each_nth_chunk)
    test_dataset = time_series_to_dataset(test_ts, chunk_len=chunk_len, reverse=True,
                                          take_each_nth_chunk=take_each_nth_chunk)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=loader_batch_size)
    backward = DataHolderOneDirection(train_dataset, test_dataset, train_loader)

    return AllDataHolder(forward, backward, test_ts=test_ts, train_ts=train_ts)


def test_chop_time_series_into_chunks() -> None:
    def compare(actual: NDArray, expected: NDArray) -> None:
        assert np.array_equal(actual, expected), f"{actual} \n\t!=\n{expected}"

    simple_data = np.array([[1], [2], [3], [4]])
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

    compare(
        chop_time_series_into_chunks(simple_data, chunk_len=1, take_each_nth_chunk=1),
        np.array([[[1]], [[2]], [[3]], [[4]]])
    )

    compare(
        chop_time_series_into_chunks(simple_data, chunk_len=1, take_each_nth_chunk=3),
        np.array([[[1]], [[4]]])
    )

    compare(
        chop_time_series_into_chunks(simple_data, chunk_len=2, take_each_nth_chunk=1),
        np.array([[[1], [2]], [[2], [3]], [[3], [4]]])
    )

    compare(
        chop_time_series_into_chunks(simple_data, chunk_len=3, take_each_nth_chunk=1),
        np.array([[[1], [2], [3]], [[2], [3], [4]]])
    )

    compare(
        chop_time_series_into_chunks(data, chunk_len=1, take_each_nth_chunk=1),
        np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]], [[10, 11, 12]], [[13, 14, 15]]])
    )

    compare(
        chop_time_series_into_chunks(data, chunk_len=2, take_each_nth_chunk=2),
        np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    )

    compare(
        chop_time_series_into_chunks(data, chunk_len=2, take_each_nth_chunk=3),
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
