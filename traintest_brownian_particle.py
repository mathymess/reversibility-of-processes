import functools
import torch
import numpy as np
import numpy.typing
from typing import Optional, Dict

import tqdm

from brownian_datagen import BrownianDatagen
from datasets import TimeSeriesDataset, DataHolderOneDirection, AllDataHolder
from models import ThreeFullyConnectedLayers
from train_test_utils import train_loop_adam_with_scheduler, EpochlyCallbackBare, write_json_to_file

NDArray = numpy.typing.NDArray[np.floating]


def create_brownian_particle_alldataholder(brownian: BrownianDatagen,
                                           window_len: int,
                                           numParticles: int = 50,
                                           loader_batch_size: int = 20,
                                           test_same_as_train: bool = True,
                                           rng_seed: Optional[int] = 42) -> AllDataHolder:
    generate_windows_targets = functools.partial(brownian.windows_targets,
                                                 window_len=window_len,
                                                 numParticles=numParticles)

    windows_f, targets_f = generate_windows_targets(rng_seed=rng_seed)
    windows_b, targets_b = generate_windows_targets(rng_seed=None, backward=True)

    train_f = TimeSeriesDataset(np.expand_dims(windows_f, -1), np.expand_dims(targets_f, -1))
    train_b = TimeSeriesDataset(np.expand_dims(windows_b, -1), np.expand_dims(targets_b, -1))

    if test_same_as_train:
        forward = DataHolderOneDirection(train_f, train_f, torch.utils.data.DataLoader(train_f))
        backward = DataHolderOneDirection(train_b, train_b, torch.utils.data.DataLoader(train_b))
    else:
        test_w_f, test_t_f = generate_windows_targets(rng_seed=None)
        test_w_b, test_t_b = generate_windows_targets(rng_seed=None, backward=True)

        test_f = TimeSeriesDataset(np.expand_dims(test_w_f, -1), np.expand_dims(test_t_f, -1))
        test_b = TimeSeriesDataset(np.expand_dims(test_w_b, -1), np.expand_dims(test_t_b, -1))

        forward = DataHolderOneDirection(train_f, test_f, torch.utils.data.DataLoader(train_f))
        backward = DataHolderOneDirection(train_b, test_b, torch.utils.data.DataLoader(train_b))

    return AllDataHolder(forward, backward)


def train_test_distribution_with_dh(dh: AllDataHolder,
                                    hidden_size: int = 13,
                                    num_epochs: int = 50,
                                    num_runs: int = 100,
                                    save_output_to_file: str = "") -> Dict:
    train_loop = functools.partial(train_loop_adam_with_scheduler, num_epochs=num_epochs)

    def get_model() -> ThreeFullyConnectedLayers:
        return ThreeFullyConnectedLayers(window_len=len(dh.forward.train_dataset.windows[0]),
                                         target_len=1,
                                         datapoint_size=1,
                                         hidden_layer1_size=hidden_size,
                                         hidden_layer2_size=hidden_size)

    def get_forward_losses():
        m = get_model()
        callback = EpochlyCallbackBare()
        train_loop(m, dh.forward.train_loader, dh.forward.test_dataset, epochly_callback=callback)
        return callback.get_values()

    def get_backward_losses():
        m = get_model()
        callback = EpochlyCallbackBare()
        train_loop(m, dh.backward.train_loader, dh.backward.test_dataset, epochly_callback=callback)
        return callback.get_values()

    forward_losses, backward_losses = [], []
    for _ in tqdm.trange(num_runs):
        forward_losses.append(get_forward_losses())
        backward_losses.append(get_backward_losses())

    result = {"forward": forward_losses, "backward": backward_losses}
    write_json_to_file(result, save_output_to_file)

    return result


if __name__ == "__main__":
    b = BrownianDatagen(kBT=1., γ=1., k=1., λ_τ=5., τ=10.)
    # dh = create_brownian_particle_alldataholder(b, window_len=5, numParticles=10)
    # print(len(dh.forward.train_dataset))
    # train_test_distribution_with_dh(dh, num_epochs=30, num_runs=20, hidden_size=3,
    #                                 save_output_to_file="20230626_distributions/brownian/03.json")

    # dh = create_brownian_particle_alldataholder(b, window_len=5, numParticles=10, test_same_as_train=False)
    # train_test_distribution_with_dh(dh, num_epochs=20, num_runs=10, hidden_size=3,
    #                                 save_output_to_file="20230626_distributions/brownian/04.json")

    # dh = create_brownian_particle_alldataholder(b, window_len=5, numParticles=10, test_same_as_train=False)
    # train_test_distribution_with_dh(dh, num_epochs=20, num_runs=40, hidden_size=5,
    #                                 save_output_to_file="20230626_distributions/brownian/05.json")

    dh = create_brownian_particle_alldataholder(b, window_len=2, numParticles=40, test_same_as_train=False)
    train_test_distribution_with_dh(dh, num_epochs=20, num_runs=40, hidden_size=5,
                                    save_output_to_file="20230626_distributions/brownian/06.json")