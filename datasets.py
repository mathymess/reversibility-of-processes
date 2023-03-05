import numpy as np
import torch

from dataclasses import dataclass
from typing import Optional
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]


def train_test_split(time_series: NDArray,
                     train_test_ratio: float = 0.7,
                     shift: int | float = 0) -> tuple[NDArray, NDArray]:
    assert time_series.ndim == 2, "Time series expected, each datapoint is a 1D array"
    assert 0 < train_test_ratio < 1, "train_test_ratio should be within (0, 1) interval"

    if isinstance(shift, float):
        assert .0 <= shift < 1., "if 'shift' argument is a float, it should be from [0,1)"
        shift = int(len(time_series) * shift)

    split_index = int(len(time_series) * train_test_ratio)
    time_series = np.roll(time_series, shift=shift)
    train, test = time_series[:split_index], time_series[split_index:]

    assert len(train) > 0, "Bad train-test split"
    assert len(test) > 0, "Bad train-test split"

    return train, test


def chop_time_series_into_chunks(time_series: NDArray,
                                 chunk_len: int,
                                 take_each_nth_chunk: int,
                                 reverse: bool = False) -> NDArray:
    assert time_series.ndim == 2, "Time series expected, each datapoint is a 1D array"
    assert len(time_series) >= chunk_len, f"chunk_len={chunk_len} is too large"

    if reverse:
        time_series = np.flip(time_series, axis=0)

    chunks = np.array([time_series[i:i+chunk_len]
                      for i in range(0, len(time_series) - chunk_len + 1, take_each_nth_chunk)])

    return chunks


def split_chunks_into_windows_and_targets(chunks: NDArray,
                                          target_len: int = 1) -> tuple[NDArray, NDArray]:
    assert chunks.ndim == 3, "Shape should be (n_chunks, chunk_len, datapoint_dim)"

    chunk_len: int = chunks.shape[1]
    assert 0 < target_len < chunk_len, f"target_len={target_len} is too large or non-positive"

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
    chunks = chop_time_series_into_chunks(time_series=ts,
                                          chunk_len=chunk_len,
                                          take_each_nth_chunk=take_each_nth_chunk,
                                          reverse=reverse)
    windows, targets = split_chunks_into_windows_and_targets(chunks)
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
                                     loader_batch_size: int = 20,
                                     take_each_nth_chunk: Optional[int] = None) -> AllDataHolder:
    assert chunk_len >= 2
    assert loader_batch_size >= 1

    if take_each_nth_chunk is None:
        take_each_nth_chunk = int(chunk_len * 0.2)

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


def test_train_test_split():
    def compare(actual: tuple[NDArray, NDArray],
                expected: tuple[NDArray, NDArray]) -> None:
        assert len(actual) == len(expected) == 2
        assert np.array_equal(actual[0], expected[0]), f"{actual} \n\t!=\n{expected}"
        assert np.array_equal(actual[1], expected[1]), f"{actual} \n\t!=\n{expected}"

    simple_data = np.array([[1], [2], [3], [4]])
    compare(
        train_test_split(simple_data, train_test_ratio=0.5),
        (np.array([[1], [2]]), np.array([[3], [4]]))
    )
    compare(
        train_test_split(simple_data, train_test_ratio=0.5, shift=1),
        (np.array([[4], [1]]), np.array([[2], [3]]))
    )
    print("Tests for train_test_split passed successfully")


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
        chop_time_series_into_chunks(simple_data, chunk_len=2,
                                     take_each_nth_chunk=1, reverse=True),
        np.array([[[4], [3]], [[3], [2]], [[2], [1]]])
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

    print("Tests for chop_time_series_into_chunks passed successfully")


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

    print("Tests for split_chunks_into_windows_and_targets passed successfully")


if __name__ == "__main__":
    test_train_test_split()
    test_chop_time_series_into_chunks()
    test_split_chunks_into_windows_and_targets()
