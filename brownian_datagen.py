from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing

from datasets import chop_time_series_into_chunks, split_chunks_into_windows_and_targets


NDArray = numpy.typing.NDArray[np.floating]


class BrownianDatagen:
    def __init__(self, kBT: float = 0.03, γ: float = 3., k: float = 2.,
                 λ_τ: float = 1, τ: float = 1):
        self.kBT = kBT  # Temperature
        self.γ = γ  # Damping rate
        self.k = k  # Strength of the potential
        self.λ_τ = λ_τ  # Centre of the potential moves from 0 to λ_τ
        self.τ = τ  # Time frame is from 0 to τ

    def generate(self, numParticles: int = 1000, numSteps: int = 99,
                 rng_seed: Optional[int] = 42, backward: bool = False) -> NDArray:
        if rng_seed is not None:
            np.random.seed(rng_seed)

        δt = self.τ / (numSteps + 1)
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
        return xs[:, ::-1] if backward else xs

    def windows_targets(self,
                        window_len: int,
                        backward: bool = False, **kwargs) -> Tuple[NDArray, NDArray]:
        traj = self.generate(backward=backward, **kwargs)
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


if __name__ == "__main__":
    def plot_trajectories(traj: NDArray, title: Optional[str] = None) -> None:
        for p in traj[:10]:
            plt.plot(p, "o-", linewidth=1, alpha=0.7)

        plt.ylabel("coordinate of the particle")
        plt.xlabel("index")
        if title is not None:
            plt.title(title)

        plt.grid()
        plt.show()

    b = BrownianDatagen(kBT=0.03, k=3., γ=2.)
    plot_trajectories(b.generate())
    plot_trajectories(b.generate(backward=True))

    w, t = b.windows_targets(3, numParticles=50)
    print("windows.shape=", w.shape, "targets.shape=", t.shape)
