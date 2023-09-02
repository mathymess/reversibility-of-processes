import os
import torch
import matplotlib.pyplot as plt

from generate_time_series import load_logistic_map_time_series
from bayesian import (plot_predictions,
                      quality_metrics,
                      BayesTrainData,
                      ExperimentResults,
                      posterior_predictive_forward_and_backward)
from bayesian_varinf import (ExpResultsWithTwoLosses,)
                             # posterior_predictive_forward_and_backward)


if __name__ == "__main__":
    d = BayesTrainData(load_logistic_map_time_series(1500),
                       window_len=1, noise_std=0.02)

    save_dir = "20230724_preds/debug_varinf/logistic_nonvarinf"

    if not os.path.isdir(save_dir):
        posterior_predictive_forward_and_backward(train_d=d, save_dir=save_dir, num_samples=60)
    else:
        print(f"Directory '{save_dir}' exists, won't overwrite")

    x_test = torch.linspace(-0.2, 1.2, 1000).reshape(-1, 1)
    y_test = (4.0 * x_test * (1 - x_test))

    er = ExperimentResults(save_dir)
    preds = er.predictive_f(x_test)["obs"]

    plot_predictions(true=y_test, pred_all=preds)
    plt.show()
