from generate_time_series import load_two_body_problem_time_series
from datasets import AllDataHolder, prepare_time_series_for_learning
from models import ThreeFullyConnectedLayers
from train_test_utils import EpochlyCallback, train_loop_adam_with_scheduler

from typing import Callable, Iterable


def load_two_body_problem_dataholder(window_len: int,
                                     target_len: int) -> AllDataHolder:
    twb = load_two_body_problem_time_series()
    # Use the same data for train and test.
    dh = prepare_time_series_for_learning(train_ts=twb,
                                          test_ts=twb.copy(),
                                          window_len=window_len,
                                          target_len=target_len,
                                          take_each_nth_chunk=1)
    return dh


def train_test_kepler(window_len: int,
                      target_len: int,
                      train_loop: Callable = train_loop_adam_with_scheduler,
                      hidden_layer1_size: int = 9,
                      hidden_layer2_size: int = 9,
                      num_epochs: int = 50,
                      tensorboard_scalar_name: str = "mean_loss_on_test") -> None:
    dh = load_two_body_problem_dataholder(window_len=window_len, target_len=target_len)

    # Forward
    forward_model = ThreeFullyConnectedLayers(window_len=window_len,
                                              target_len=target_len,
                                              datapoint_size=2,
                                              hidden_layer1_size=hidden_layer1_size,
                                              hidden_layer2_size=hidden_layer2_size)
    forward_callback = EpochlyCallback(
        tensorboard_log_dir="runs/20230320_kepler_window/forward/",
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
        tensorboard_log_dir="runs/20230320_kepler_window/backward/",
        tensorboard_scalar_name=tensorboard_scalar_name)
    train_loop(backward_model,
               dh.backward.train_loader,
               dh.backward.test_dataset,
               num_epochs=num_epochs,
               epochly_callback=backward_callback)


if __name__ == "__main__":
    def get_target_len_mesh(window_len: int) -> Iterable[int]:
        return range(1, window_len, max(window_len // 5, 1))

    for window_len in range(1, 100, 5):
        for target_len in get_target_len_mesh(window_len):
            for attempt in "abc":
                scalar_name = f"window={window_len}/target={target_len}_{attempt}"
                train_test_kepler(window_len=window_len,
                                  target_len=target_len,
                                  tensorboard_scalar_name=scalar_name)
