from generate_time_series import (load_logistic_map_time_series,
                                  load_henon_map_time_series,
                                  load_garch_time_series)
from train_test_utils import train_test_distribution, train_test_distribution_montecarlo_ts

import numpy as np


def draft_traintest_logistic_map():
    log = load_logistic_map_time_series(length=3000, coef_a=4, x_initial=0.6)
    filepath = "20230626_distributions/logistic_window=4_x_initial=0.6_length=3000.json"
    train_test_distribution(log, num_runs=1,
                            window_len=4, hidden_size=20, num_epochs=50,
                            save_output_to_file=filepath)

    np.random.seed(42)
    collection = [load_logistic_map_time_series(length=1500, x_initial=x) for x
                  in np.random.uniform(0, 1, size=3)]
    filepath = "20230626_distributions/logistic_montecarlo_window=4_length=1500.json"
    train_test_distribution_montecarlo_ts(collection,
                                          window_len=4, hidden_size=15,
                                          datapoint_size=1, num_epochs=30,
                                          save_output_to_file=filepath)


def draft_traintest_henon_map():
    hen = load_henon_map_time_series(length=2000)
    filepath = "20230626_distributions/henon_length=2000.json"
    train_test_distribution(hen, num_runs=1,
                            window_len=7, hidden_size=20, num_epochs=50,
                            save_output_to_file=filepath)

    np.random.seed(45)
    collection = [load_henon_map_time_series(length=1500, x_initial=x, y_initial=y) for x, y
                  in np.random.uniform(0, 1, size=6).reshape(-1, 2)]
    filepath = "20230626_distributions/henon_montecarlo_length=1500.json"
    train_test_distribution_montecarlo_ts(collection,
                                          window_len=7, hidden_size=15,
                                          datapoint_size=2, num_epochs=30,
                                          save_output_to_file=filepath)


def draft_traintest_garch():
    np.random.seed(47)
    grc = load_garch_time_series(length=2100)
    filepath = "20230626_distributions/garch_length=2100.json"
    train_test_distribution(grc, num_runs=1,
                            window_len=7, hidden_size=20, num_epochs=10,
                            save_output_to_file=filepath)

    collection = [load_garch_time_series(length=2000, x_initial=x) for x
                  in np.random.uniform(0, 1, size=6).reshape(-1, 3)]
    filepath = "20230626_distributions/garch_montecarlo_length=2000.json"
    train_test_distribution_montecarlo_ts(collection,
                                          window_len=7, hidden_size=20,
                                          datapoint_size=1, num_epochs=10,
                                          save_output_to_file=filepath)


if __name__ == "__main__":
    # draft_traintest_logistic_map()
    # draft_traintest_henon_map()
    draft_traintest_garch()
