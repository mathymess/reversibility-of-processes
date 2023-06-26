from generate_time_series import load_logistic_map_time_series
from train_test_utils import train_test_distribution, train_test_distribution_montecarlo_ts

import numpy as np

if __name__ == "__main__":
    log = load_logistic_map_time_series(length=3000, coef_a=4, x_initial=0.6)
    filepath = "20230626_distributions/logistic_coef_a=4_x_initial=0.6_length=3000.json"
    train_test_distribution(log, num_runs=1,
                            window_len=7, hidden_size=20, num_epochs=50,
                            save_output_to_file=filepath)

    np.random.seed(42)
    collection = [load_logistic_map_time_series(length=1500, x_initial=x) for x
                  in np.random.uniform(0, 1, size=3)]
    filepath = "20230626_distributions/logistic_montecarlo_coef_a=4_length=1500.json"
    train_test_distribution_montecarlo_ts(collection,
                                          window_len=7, hidden_size=15,
                                          datapoint_size=1, num_epochs=30,
                                          save_output_to_file=filepath)
