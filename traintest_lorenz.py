from generate_time_series import load_lorenz_attractor_time_series
from datasets import AllDataHolder, prepare_time_series_for_learning
from models import ThreeFullyConnectedLayers
from train_test_utils import EpochlyCallback, train_loop


def load_lorenz_attractor_dataholder(window_len: int,
                                     target_len: int) -> AllDataHolder:
    lrz = load_lorenz_attractor_time_series()
    # Use the same data for train and test.
    dh = prepare_time_series_for_learning(train_ts=lrz,
                                          test_ts=lrz.copy(),
                                          window_len=window_len,
                                          target_len=target_len,
                                          take_each_nth_chunk=1)
    return dh


def train_test_lorenz(window_len: int = 50,
                      target_len: int = 1,
                      hidden_layer1_size: int = 100,
                      hidden_layer2_size: int = 100,
                      num_epochs: int = 50,
                      tensorboard_scalar_name: str = "mean_loss_on_test") -> None:
    dh = load_lorenz_attractor_dataholder(window_len=window_len, target_len=target_len)

    # Forward
    forward_model = ThreeFullyConnectedLayers(window_len=window_len,
                                              target_len=target_len,
                                              datapoint_size=3,
                                              hidden_layer1_size=hidden_layer1_size,
                                              hidden_layer2_size=hidden_layer2_size)
    forward_callback = EpochlyCallback(tensorboard_log_dir="runs/20230313_lorenz/forward/",
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
    backward_callback = EpochlyCallback(tensorboard_log_dir="runs/20230313_lorenz/backward/",
                                        tensorboard_scalar_name=tensorboard_scalar_name)
    train_loop(backward_model,
               dh.backward.train_loader,
               dh.backward.test_dataset,
               num_epochs=num_epochs,
               epochly_callback=backward_callback)


if __name__ == "__main__":
    for size in range(1, 20):
        for attempt in "abc":
            scalar_name = f"hidden_layer_size/{size}{attempt}"
            train_test_lorenz(window_len=30,
                              hidden_layer1_size=size,
                              hidden_layer2_size=size,
                              tensorboard_scalar_name=scalar_name)
