from generate_time_series import load_double_pendulum_time_series
from datasets import AllDataHolder, prepare_time_series_for_learning
from models import ThreeFullyConnectedLayers
from train_test_utils import EpochlyCallback, train_loop_adam_with_scheduler

from typing import Callable


def load_double_pendulum_dataholder(window_len: int,
                                    target_len: int) -> AllDataHolder:
    dbp = load_double_pendulum_time_series()
    # Use the same data for train and test.
    dh = prepare_time_series_for_learning(train_ts=dbp,
                                          test_ts=dbp.copy(),
                                          window_len=window_len,
                                          target_len=target_len,
                                          take_each_nth_chunk=1)
    return dh


def train_test_double_pendulum(window_len: int,
                               target_len: int = 1,
                               train_loop: Callable = train_loop_adam_with_scheduler,
                               hidden_layer1_size: int = 13,
                               hidden_layer2_size: int = 13,
                               num_epochs: int = 50,
                               tensorboard_scalar_name: str = "mean_loss_on_test") -> None:
    dh = load_double_pendulum_dataholder(window_len=window_len, target_len=target_len)

    # Forward
    forward_model = ThreeFullyConnectedLayers(window_len=window_len,
                                              target_len=target_len,
                                              datapoint_size=2,
                                              hidden_layer1_size=hidden_layer1_size,
                                              hidden_layer2_size=hidden_layer2_size)
    forward_callback = EpochlyCallback(
            tensorboard_log_dir="runs/20230428_double_pendulum/forward/",
            tensorboard_scalar_name=tensorboard_scalar_name)
    train_loop(forward_model,
               dh.forward.train_loader,
               dh.forward.test_dataset,
               num_epochs=num_epochs,
               epochly_callback=forward_callback)

    # Backward
    backward_model = ThreeFullyConnectedLayers(window_len=window_len,
                                               target_len=target_len,
                                               datapoint_size=2,
                                               hidden_layer1_size=hidden_layer1_size,
                                               hidden_layer2_size=hidden_layer2_size)
    backward_callback = EpochlyCallback(
            tensorboard_log_dir="runs/20230428_double_pendulum/backward/",
            tensorboard_scalar_name=tensorboard_scalar_name)
    train_loop(backward_model,
               dh.backward.train_loader,
               dh.backward.test_dataset,
               num_epochs=num_epochs,
               epochly_callback=backward_callback)


if __name__ == "__main__":
    for window_len in (5, 12, 25):
        for size in (10, 20):
            for attempt in "abcd":
                scalar_name = f"window_len={window_len}|size={size}/{attempt}"
                train_test_double_pendulum(window_len=window_len,
                                           hidden_layer1_size=size,
                                           hidden_layer2_size=size,
                                           tensorboard_scalar_name=scalar_name)
