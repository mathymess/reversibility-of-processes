from generate_time_series import load_harmonic_oscillator_time_series
from datasets import AllDataHolder, prepare_time_series_for_learning
from models import ThreeFullyConnectedLayers
from train_test_utils import (EpochlyCallback,
                              train_loop_adam_with_scheduler,
                              train_test_distribution)

from typing import Callable


def load_harmonic_oscillator_dataholder(window_len: int,
                                        target_len: int,
                                        friction: float = 0.05) -> AllDataHolder:
    hos = load_harmonic_oscillator_time_series(friction=friction)
    # Use the same data for train and test.
    dh = prepare_time_series_for_learning(train_ts=hos,
                                          test_ts=hos.copy(),
                                          window_len=window_len,
                                          target_len=target_len,
                                          take_each_nth_chunk=1)
    return dh


def train_test_harmonic_oscillator(window_len: int = 7,
                                   target_len: int = 1,
                                   train_loop: Callable = train_loop_adam_with_scheduler,
                                   hidden_layer1_size: int = 7,
                                   hidden_layer2_size: int = 7,
                                   num_epochs: int = 20,
                                   tensorboard_scalar_name: str =
                                   "damped harmonic oscillator") -> None:
    dh = load_harmonic_oscillator_dataholder(window_len=window_len, target_len=target_len)

    # Forward
    forward_model = ThreeFullyConnectedLayers(window_len=window_len,
                                              target_len=target_len,
                                              datapoint_size=1,
                                              hidden_layer1_size=hidden_layer1_size,
                                              hidden_layer2_size=hidden_layer2_size)
    forward_callback = EpochlyCallback(
        tensorboard_log_dir="runs/20230510_hos/forward/",
        tensorboard_scalar_name=tensorboard_scalar_name)
    train_loop(forward_model,
               dh.forward.train_loader,
               dh.forward.test_dataset,
               num_epochs=num_epochs,
               epochly_callback=forward_callback)

    # Backward
    backward_model = ThreeFullyConnectedLayers(window_len=window_len,
                                               target_len=target_len,
                                               datapoint_size=1,
                                               hidden_layer1_size=hidden_layer1_size,
                                               hidden_layer2_size=hidden_layer2_size)
    backward_callback = EpochlyCallback(
        tensorboard_log_dir="runs/20230510_hos/backward/",
        tensorboard_scalar_name=tensorboard_scalar_name)
    train_loop(backward_model,
               dh.backward.train_loader,
               dh.backward.test_dataset,
               num_epochs=num_epochs,
               epochly_callback=backward_callback)

    print(forward_callback.all_values)
    print(backward_callback.all_values)


if __name__ == "__main__":
    hos = load_harmonic_oscillator_time_series(friction=0.02, t_duration=300)
    name_prefix = "20230507_distributions/damped_harmonic_oscillator_"

    train_test_distribution(hos, window_len=7, hidden_size=7, num_epochs=20,
                            save_output_to_file=name_prefix + "size=7_window_len=7.json")

    train_test_distribution(hos, window_len=7, hidden_size=13, num_epochs=30,
                            save_output_to_file=name_prefix + "size=13_window_len=7.json")
