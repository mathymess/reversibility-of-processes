from generate_time_series import (load_lorenz_attractor_time_series,
                                  load_two_body_problem_time_series,
                                  load_double_pendulum_time_series)

from scipy.spatial.distance import jensenshannon
# from scipy.stats import entropy
import matplotlib.pyplot as plt

import itertools
import numpy as np
from typing import Dict, Tuple
import numpy.typing
NDArray = numpy.typing.NDArray[np.floating]
FreqDict = Dict[Tuple[float, ...], float]


def permutation_distribution(time_series: NDArray, embed_dim: int = 5) -> Dict:
    if time_series.ndim != 1:
        raise ValueError("Only 1D time series are suitable for the permutation test")

    freqs = {p: 0. for p in itertools.permutations(range(embed_dim))}
    for k in range(len(time_series) - embed_dim + 1):
        window = time_series[k: k + embed_dim]
        freqs[tuple(np.argsort(window))] += 1.

    norm_factor = sum(freqs.values())
    for key in freqs.keys():
        freqs[key] /= norm_factor

    return freqs


def plot_permutation_distribution(freq: FreqDict):
    plt.bar(freq.values())
    plt.show()


def freq_dicts_to_prob_arrays(freq1: FreqDict, freq2: FreqDict):
    """Put values from 2 dicts with same keys into 2 correspondingly ordered lists"""
    assert freq1.keys() == freq2.keys()

    prob1, prob2 = [], []
    for k in freq1.keys():
        prob1.append(freq1[k])
        prob2.append(freq2[k])

    return prob1, prob2


def time_asymmetry_metric(time_series: NDArray, embed_dim: int = 5) -> float:
    freqs_forward = permutation_distribution(time_series, embed_dim=embed_dim)
    freqs_backward = permutation_distribution(np.flip(time_series), embed_dim=embed_dim)
    return jensenshannon(*freq_dicts_to_prob_arrays(freqs_forward, freqs_backward))


def test_trivia():
    time_series = np.array([0, 1, 1, 0, 3, 4, 5])
    d = permutation_distribution(time_series, embed_dim=3)
    print(d)
    plt.bar(range(len(d)), list(d.values()), align='center')
    plt.show()

    print(time_asymmetry_metric(time_series))


def test_our_systems():
    lrz = time_asymmetry_metric(load_lorenz_attractor_time_series()[:3200, 0])
    twb = time_asymmetry_metric(load_two_body_problem_time_series()[:3200, 0])
    dbp = time_asymmetry_metric(load_double_pendulum_time_series()[:3200, 0])
    print(f"x: lrz={lrz}, twb={twb}, dbp={dbp}")

    lrz = time_asymmetry_metric(load_lorenz_attractor_time_series()[:3200, 1])
    twb = time_asymmetry_metric(load_two_body_problem_time_series()[:3200, 1])
    dbp = time_asymmetry_metric(load_double_pendulum_time_series()[:3200, 1])
    print(f"y: lrz={lrz}, twb={twb}, dbp={dbp}")


if __name__ == "__main__":
    test_our_systems()
