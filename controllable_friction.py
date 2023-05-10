from generate_time_series import load_double_pendulum_time_series
from train_test_utils import train_test_distribution

import numpy as np


def calculate_distribution(friction: float,
                           hidden_layer_size: int = 20,
                           window_len: int = 15) -> None:
    time_series = load_double_pendulum_time_series(friction=friction)
    filepath = (f"20230507_distributions/doublependulum_friction={friction}_"
                f"size={hidden_layer_size}_window_len={window_len}.json")
    train_test_distribution(time_series,
                            window_len=window_len,
                            save_output_to_file=filepath,
                            hidden_size=hidden_layer_size,
                            num_epochs=30)


def main() -> None:
    for friction in np.linspace(0.003, 0.1, 10):
        calculate_distribution(friction)


if __name__ == "__main__":
    main()
