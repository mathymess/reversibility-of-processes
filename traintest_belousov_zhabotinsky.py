from generate_time_series import load_belousov_zhabotinsky_time_series
from datasets import train_test_split, AllDataHolder, prepare_time_series_for_learning
from models import ThreeFullyConnectedLayers
from train_test_utils import EpochlyCallback, train_loop

import itertools


def load_belousov_zhabotinsky_dataholder(window_len: int,
                                         target_len: int,
                                         shift_ratio: float) -> AllDataHolder:
    assert 0 <= shift_ratio < 1
    bzh = load_belousov_zhabotinsky_time_series()
    bzh_train, bzh_test = train_test_split(bzh, shift=shift_ratio)
    dh = prepare_time_series_for_learning(bzh_train, bzh_test,
                                          window_len=window_len, target_len=target_len)
    return dh


def train_test_belousov_zhabotinsky(window_len: int = 50,
                                    target_len: int = 1,
                                    shift_ratio: float = 0.,
                                    hidden_layer1_size: int = 100,
                                    hidden_layer2_size: int = 100,
                                    num_epochs: int = 30,
                                    tensorboard_scalar_name: str = "mean_loss_on_test") -> None:
    dh = load_belousov_zhabotinsky_dataholder(window_len=window_len,
                                              target_len=target_len,
                                              shift_ratio=shift_ratio)

    # Forward
    forward_model = ThreeFullyConnectedLayers(window_len=window_len,
                                              target_len=target_len,
                                              datapoint_size=3,
                                              hidden_layer1_size=hidden_layer1_size,
                                              hidden_layer2_size=hidden_layer2_size)
    forward_callback = EpochlyCallback(
            tensorboard_log_dir="runs/20230306_belousov_zhabotinsky/forward/",
            tensorboard_scalar_name=tensorboard_scalar_name)
    train_loop(forward_model,
               dh.forward.train_loader,
               dh.forward.test_dataset,
               num_epochs=num_epochs,
               epochly_callback=forward_callback)

    # Backward
    backward_model = ThreeFullyConnectedLayers(window_len=window_len,
                                               target_len=target_len,
                                               datapoint_size=3,
                                               hidden_layer1_size=hidden_layer1_size,
                                               hidden_layer2_size=hidden_layer2_size)
    backward_callback = EpochlyCallback(
            tensorboard_log_dir="runs/20230306_belousov_zhabotinsky/backward/",
            tensorboard_scalar_name=tensorboard_scalar_name)
    train_loop(backward_model,
               dh.backward.train_loader,
               dh.backward.test_dataset,
               num_epochs=num_epochs,
               epochly_callback=backward_callback)


if __name__ == "__main__":
    for window_len, shift_ratio in itertools.product((10, 30, 70, 110), (.1, .4, .6, .9)):
        scalar_name = f"window_len:shift_ratio/{window_len}+{shift_ratio}"
        train_test_belousov_zhabotinsky(window_len=window_len,
                                        shift_ratio=shift_ratio,
                                        tensorboard_scalar_name=scalar_name)

    for size in (300, 250, 200, 150, 110, 80, 60, 40, 25, 15):
        scalar_name = f"hidden_layer_size_at_window50/{size}"
        train_test_belousov_zhabotinsky(window_len=50,
                                        hidden_layer1_size=size,
                                        hidden_layer2_size=size,
                                        tensorboard_scalar_name=scalar_name,
                                        num_epochs=int(size/2))
