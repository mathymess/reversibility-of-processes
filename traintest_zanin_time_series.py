from generate_time_series import (load_logistic_map_time_series,
                                  load_henon_map_time_series,
                                  load_arnold_map_time_series,
                                  load_arch_time_series,
                                  load_garch_time_series)
from train_test_utils import train_test_distribution, train_test_distribution_montecarlo_ts

import numpy as np
import tqdm


def traintest_logistic_map():
    log = load_logistic_map_time_series(length=3000, coef_a=4, x_initial=0.6)
    filepath = "20230626_distributions/logistic.json"
    train_test_distribution(log, num_runs=100,
                            window_len=5, hidden_size=20, num_epochs=30,
                            save_output_to_file=filepath)
    print(filepath)

    np.random.seed(42)
    collection = [load_logistic_map_time_series(length=1500, x_initial=x) for x
                  in np.random.uniform(0, 1, size=100)]
    filepath = "20230626_distributions/logistic_montecarlo.json"
    train_test_distribution_montecarlo_ts(collection,
                                          window_len=5, hidden_size=15,
                                          datapoint_size=1, num_epochs=30,
                                          save_output_to_file=filepath)
    print(filepath)


def traintest_henon_map():
    hen = load_henon_map_time_series(length=2000)
    filepath = "20230626_distributions/henon.json"
    train_test_distribution(hen, num_runs=100,
                            window_len=5, hidden_size=20, num_epochs=30,
                            save_output_to_file=filepath)
    print(filepath)

    np.random.seed(45)
    collection = [load_henon_map_time_series(length=1500, x_initial=x, y_initial=y) for x, y
                  in np.random.uniform(0.1, 0.9, size=200).reshape(-1, 2)]
    filepath = "20230626_distributions/henon_montecarlo.json"
    train_test_distribution_montecarlo_ts(collection,
                                          window_len=5, hidden_size=20,
                                          datapoint_size=2, num_epochs=30,
                                          save_output_to_file=filepath)
    print(filepath)


def traintest_arnold_map():
    arn = load_arnold_map_time_series(length=2000)
    filepath = "20230626_distributions/arnold.json"
    train_test_distribution(arn, num_runs=100,
                            window_len=5, hidden_size=20, num_epochs=30,
                            save_output_to_file=filepath)
    print(filepath)

    np.random.seed(45)
    collection = [load_arnold_map_time_series(length=2000, x_initial=x, y_initial=y) for x, y
                  in np.random.uniform(0, 1, size=200).reshape(-1, 2)]
    filepath = "20230626_distributions/arnold_montecarlo.json"
    train_test_distribution_montecarlo_ts(collection,
                                          window_len=5, hidden_size=20,
                                          datapoint_size=2, num_epochs=30,
                                          save_output_to_file=filepath)
    print(filepath)


def traintest_arch():
    np.random.seed(47)
    grc = load_arch_time_series(length=3000)
    filepath = "20230626_distributions/arch.json"
    train_test_distribution(grc, num_runs=100,
                            window_len=5, hidden_size=20, num_epochs=10,
                            save_output_to_file=filepath)
    print(filepath)

    collection = [load_arch_time_series(length=3000, x_initial=x) for x
                  in np.random.uniform(0, 1, size=300).reshape(-1, 3)]
    filepath = "20230626_distributions/arch_montecarlo.json"
    train_test_distribution_montecarlo_ts(collection,
                                          window_len=5, hidden_size=20,
                                          datapoint_size=1, num_epochs=10,
                                          save_output_to_file=filepath)
    print(filepath)


def traintest_garch():
    np.random.seed(48)
    grc = load_garch_time_series(length=3000)
    filepath = "20230626_distributions/garch_window_len=20.json"
    train_test_distribution(grc, num_runs=10,
                            window_len=20, hidden_size=20, num_epochs=30,
                            save_output_to_file=filepath)
    print(filepath)


def traintest_logistic_vs_length():
    np.random.seed(154125)
    for length in tqdm.tqdm((650, 850, 1150, 1500, 1850, 2250, 2750, 3250, 3500)):
        filepath = f"20230626_distributions/logistic_vs_length/{length}.json"
        collection = [load_logistic_map_time_series(length=length, x_initial=x) for x
                      in np.random.uniform(0.6, 0.8, size=30)]
        train_test_distribution_montecarlo_ts(collection,
                                              window_len=5, hidden_size=15,
                                              datapoint_size=1, num_epochs=30,
                                              save_output_to_file=filepath)


def traintest_henon_vs_length():
    np.random.seed(154)
    for length in tqdm.tqdm((500, 650, 850, 1000, 1150, 1300, 1500, 1620, 1850,
                             2000, 2250, 2500, 2750, 3000, 3250, 3500)):
        filepath = f"20230626_distributions/henon_vs_length/{length}.json"
        collection = [load_logistic_map_time_series(length=length, x_initial=x) for x
                      in np.random.uniform(0.6, 0.8, size=30)]
        train_test_distribution_montecarlo_ts(collection,
                                              window_len=5, hidden_size=15,
                                              datapoint_size=1, num_epochs=30,
                                              save_output_to_file=filepath)


if __name__ == "__main__":
    # traintest_logistic_map()
    # traintest_henon_map()
    # traintest_arnold_map()
    # traintest_arch()
    # traintest_garch()
    traintest_logistic_vs_length()
    traintest_henon_vs_length()
