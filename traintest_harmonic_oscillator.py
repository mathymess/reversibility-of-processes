from generate_time_series import load_harmonic_oscillator_time_series
from train_test_utils import train_test_distribution

import tqdm

if __name__ == "__main__":
    for friction in tqdm.tqdm([0., 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03]):
        hos = load_harmonic_oscillator_time_series(friction=friction, t_duration=300)
        filepath = (f"20230507_distributions/damped_harmonic_oscillator_friction="
                    f"{friction}_size=13_window_len=7.json")
        train_test_distribution(hos,
                                window_len=7, hidden_size=13, num_epochs=30,
                                save_output_to_file=filepath)
