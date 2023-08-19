from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing
import functools
import torch

from bayesian import BayesTrainData
from datasets import (chop_time_series_into_chunks,
                      split_chunks_into_windows_and_targets,
                      reverse_windows_targets)


NDArray = numpy.typing.NDArray[np.floating]


class BrownianDatagen:
    def __init__(self, kBT: float = 0.03, γ: float = 3., k: float = 2.,
                 λ_τ: float = 5., τ: float = 10.):
        self.kBT = kBT  # Temperature
        self.γ = γ  # Damping rate
        self.k = k  # Strength of the potential
        self.λ_τ = λ_τ  # Centre of the potential moves from 0 to λ_τ
        self.τ = τ  # Time varies from 0 to τ

    def energy(self, x, λ):
        return 0.5 * self.k * (x - λ) ** 2

    def generate(self, numParticles: int = 1000, numSteps: int = 99,
                 rng_seed: Optional[int] = 42, backward: bool = False) -> Tuple[NDArray, NDArray]:
        if rng_seed is not None:
            np.random.seed(rng_seed)

        δt = self.τ / (numSteps + 1)
        wList = np.zeros(numParticles)

        λs = np.linspace(0, self.λ_τ, numSteps + 1)
        xs = np.zeros((numParticles, numSteps + 1))
        xs[:, 0] = self.kBT / self.k * np.random.randn(numParticles)
        if backward:
            λs = λs[::-1]
            xs[:, 0] += λs[0]

        for i in range(numSteps):
            ΔW = np.sqrt(2 * self.kBT / self.γ * δt) * np.random.randn(numParticles)
            xs[:, i+1] = (xs[:, i] * (1 - self.k / self.γ * δt)
                          + self.k / self.γ * δt * λs[i if backward else i + 1]
                          + ΔW)
            x_en = xs[:, i+1 if backward else i]
            wList += self.energy(x_en, λs[i+1]) - self.energy(x_en, λs[i])

        if backward:
            return xs[:, ::-1], -wList
        return xs, wList

    def windows_targets(self,
                        window_len: int,
                        backward: bool = False, **kwargs) -> Tuple[NDArray, NDArray]:
        traj, _ = self.generate(backward=backward, **kwargs)
        window_list = []
        target_list = []
        for pt in traj:
            chunks = chop_time_series_into_chunks(pt.reshape(-1, 1),
                                                  chunk_len=window_len+1,
                                                  take_each_nth_chunk=1)
            windows, targets = split_chunks_into_windows_and_targets(chunks)
            window_list.append(windows)
            target_list.append(targets)

        return (np.concatenate(window_list).squeeze(-1),
                np.concatenate(target_list).squeeze(-1))

    def visualize(self):
        traj_f, work_f = self.generate(numParticles=10000)
        traj_b, work_b = self.generate(numParticles=10000, backward=True)

        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5, 10))
        fig.suptitle(f"kBT={self.kBT}, γ={self.γ}, k={self.k}, λ_τ={self.λ_τ}, τ={self.τ}")

        # Plot forward trajectories
        axs[0].set_title("Forward trajectories")
        for p in traj_f[:10]:
            axs[0].plot(p, "o-", linewidth=1, markersize=1.5, alpha=0.7)
        axs[0].set_ylabel("coordinate of the particle, forward")
        axs[0].set_xlabel("index")
        axs[0].grid()

        # Plot backward trajectories
        axs[1].set_title("Backward trajectories")
        for p in traj_b[:10]:
            axs[1].plot(p, "o-", linewidth=1, markersize=1.5, alpha=0.7)
        axs[1].set_ylabel("coordinate of the particle, backward")
        axs[1].set_xlabel("index")
        axs[1].grid()

        # Plot work distributions
        axs[2].set_title("Work distributions")
        axs[2].hist(work_f, bins=100, color="red", label="forward", alpha=0.5)
        axs[2].hist(work_b, bins=100, color="blue", label="backward", alpha=0.5)
        axs[2].grid()
        axs[2].legend()

        fig.tight_layout()

        return fig, axs


def create_dataholder_for_windows_and_targets(brownian: BrownianDatagen,
                                              numParticles: int = 50,
                                              window_len: int = 3,
                                              rng_seed: Optional[int] = None) -> BayesTrainData:
    meaningless_time_series = np.zeros((5, 1), np.float32)
    dt = BayesTrainData(meaningless_time_series)
    # For Brownian motion, different windows might come from different
    # trajectories, so having one single time series doesn't make sense.
    # Morover, backward and forward don't refer to the same thing anymore:
    # they are generated separately, rather than backward = np.flip(forward).
    # But I want to use the same class for holding the train data, so I set attributes manually.
    generate_windows_targets = functools.partial(brownian.windows_targets,
                                                 window_len=window_len,
                                                 numParticles=numParticles)
    if rng_seed is not None:
        generate_windows_targets = functools.partial(generate_windows_targets, rng_seed=rng_seed)

    dt.windows_f, dt.targets_f = generate_windows_targets()
    dt.windows_b, dt.targets_b = generate_windows_targets(backward=True)
    dt.windows_f, dt.targets_f, dt.windows_b, dt.targets_b = map(
        lambda x: torch.tensor(x, dtype=torch.float32),
        (dt.windows_f, dt.targets_f, dt.windows_b, dt.targets_b))
    return dt


def reverse_windows_targets_squeezed(windows: NDArray, targets: NDArray) -> Tuple[NDArray, NDArray]:
    def unsqueeze(arr: NDArray) -> NDArray:
        return arr.reshape(*arr.shape, 1)

    windows, targets = reverse_windows_targets(unsqueeze(windows), unsqueeze(targets))
    return windows.squeeze(-1), targets.squeeze(-1)


def create_simple_flip_dataholder_for_windows_and_targets(
        brownian: BrownianDatagen,
        numParticles: int = 50,
        window_len: int = 3,
        rng_seed: Optional[int] = None) -> BayesTrainData:
    meaningless_time_series = np.zeros((5, 1), np.float32)
    dt = BayesTrainData(meaningless_time_series)
    # For Brownian motion, different windows might come from different
    # trajectories, so having one single time series doesn't make sense.
    # But I want to use the same class for holding the train data, so I set attributes manually.
    generate_windows_targets = functools.partial(brownian.windows_targets,
                                                 window_len=window_len,
                                                 numParticles=numParticles)
    if rng_seed is not None:
        generate_windows_targets = functools.partial(generate_windows_targets, rng_seed=rng_seed)

    dt.windows_f, dt.targets_f = generate_windows_targets()
    dt.windows_b, dt.targets_b = reverse_windows_targets_squeezed(dt.windows_f, dt.targets_f)

    dt.windows_f, dt.targets_f, dt.windows_b, dt.targets_b = map(
        lambda x: torch.tensor(x, dtype=torch.float32),
        (dt.windows_f, dt.targets_f, dt.windows_b, dt.targets_b))

    return dt


if __name__ == "__main__":
    b = BrownianDatagen(kBT=1., γ=4., k=5.)

    w, t = b.windows_targets(window_len=3, numParticles=50)
    print("windows.shape=", w.shape, "targets.shape=", t.shape)

    b.visualize()
    plt.show()
