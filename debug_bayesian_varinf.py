import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from generate_time_series import load_logistic_map_time_series
from bayesian import (plot_predictions,
                      quality_metrics,
                      BayesTrainData,
                      ExperimentResults,)
                      # posterior_predictive_forward_and_backward)
from bayesian_varinf import (ExpResultsWithTwoLosses,
                             posterior_predictive_forward_and_backward)


if __name__ == "__main__":
    ts = np.linspace(0., 1., 300).reshape(-1, 1)
    d = BayesTrainData(ts, window_len=1, noise_std=0.00005)

    save_dir = "20230724_preds/debug_varinf/linspace_varinf_size20/"

    if not os.path.isdir(save_dir):
        posterior_predictive_forward_and_backward(train_d=d, save_dir=save_dir,
                                                  num_samples=60,
                                                  hidden_size=20)
    else:
        print(f"Directory '{save_dir}' exists, won't overwrite")

    er = ExperimentResults(save_dir)
    preds = er.predictive_f(d.windows_f)["obs"]

    # plot_predictions(true=d.targets_f, pred_all=preds)
    plot_predictions(pred_all=preds)
    # plot_predictions(true=d.targets_f, pred_std=preds.std(0), pred_mean=preds.mean(0))
    plt.show()
