from generate_time_series import load_logistic_map_time_series
from train_test_utils import train_test_distribution

if __name__ == "__main__":
    log = load_logistic_map_time_series(length=500, coef_a=4, x_initial=0.6)
    filepath = "20230626_distributions/logistic_coef_a=4_x_initial=0.6_length=500.json"
    train_test_distribution(log, num_runs=1,
                            window_len=7, hidden_size=20, num_epochs=50,
                            save_output_to_file=filepath)
