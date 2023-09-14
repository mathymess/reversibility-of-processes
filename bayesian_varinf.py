import os
from typing import Tuple, Optional

import torch
from sklearn.model_selection import train_test_split

import pyro
from pyro.infer import Predictive

import tqdm
import tempfile

from bayesian import (save_train_data_to_drive,
                      BayesianThreeFCLayers,
                      BayesTrainData,
                      ExperimentResults,
                      quality_metrics)
from train_test_utils import write_json_to_file, LossDistribution


def get_metrics(windows: torch.Tensor,
                targets: torch.Tensor,
                model: pyro.nn.PyroModule,
                guide: pyro.infer.autoguide.AutoGuide,
                num_samples: int = 200) -> float:
    predictive = Predictive(model=model, guide=guide, num_samples=num_samples)
    preds = predictive(windows)["obs"]
    return quality_metrics(preds, targets)


TrainRetval = Tuple[Predictive, torch.Tensor, torch.Tensor]


def train_varinf(windows: torch.Tensor,
                 targets: torch.Tensor,
                 num_samples: int = 500,
                 hidden_size: int = 10,
                 num_epochs: int = 4000,
                 lr: float = 0.001,
                 prior_scale: float = 0.5,
                 train_test_split_ratio: Optional[float] = None,  # None -> Loss on train only
                 save_metrics_every_n_epochs: int = 10,
                 use_tqdm: bool = True) -> TrainRetval:
    assert windows.ndim == targets.ndim == 2
    assert windows.shape[0] == targets.shape[0]

    if train_test_split_ratio is not None:
        windows, windows_test, targets, targets_test = train_test_split(
            windows, targets, train_size=train_test_split_ratio, random_state=42)

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    model = BayesianThreeFCLayers(window_len=windows.shape[-1], target_len=1,
                                  datapoint_size=1, hidden_size=hidden_size,
                                  prior_scale=prior_scale)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

    losses = []
    metrics = []
    for epoch in tqdm.trange(num_epochs) if use_tqdm else range(num_epochs):
        loss = svi.step(windows, targets)
        losses.append(loss)

        if (epoch + 5) % save_metrics_every_n_epochs == 0:
            m = get_metrics(windows, targets, model, guide)
            if train_test_split_ratio is not None:
                m_test = get_metrics(windows_test, targets_test, model, guide)
                m = {**{"train_" + k : v for k, v in m.items()},
                     **{"test_" + k : v for k, v in m_test.items()}}
            metrics.append(m)

    losses = torch.tensor(losses, dtype=torch.float32)

    predictive = Predictive(model=model, guide=guide, num_samples=num_samples)

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    return predictive, losses, metrics


def posterior_predictive_forward_and_backward_impl(
        windows_f: torch.Tensor,
        targets_f: torch.Tensor,
        windows_b: torch.Tensor,
        targets_b: torch.Tensor,
        **kwargs) -> Tuple[TrainRetval, TrainRetval]:
    res_f = train_varinf(windows_f, targets_f, **kwargs)
    res_b = train_varinf(windows_b, targets_b, **kwargs)
    return res_f, res_b


def posterior_predictive_forward_and_backward(
        train_d: BayesTrainData,
        save_dir: str,
        **kwargs) -> Tuple[TrainRetval, TrainRetval]:
    save_train_data_to_drive(train_d, save_dir)

    res_f, res_b = posterior_predictive_forward_and_backward_impl(
        windows_f=train_d.windows_f,
        targets_f=train_d.targets_f,
        windows_b=train_d.windows_b,
        targets_b=train_d.targets_b,
        **kwargs)

    predictive_f, losses_f, metrics_f = res_f
    torch.save(predictive_f, os.path.join(save_dir, "predictive.forward.torch"))
    torch.save(losses_f, os.path.join(save_dir, "losses_f.torch"))
    torch.save(metrics_f, os.path.join(save_dir, "metrics.forward.torch"))

    predictive_b, losses_b, metrics_b = res_b
    torch.save(predictive_b, os.path.join(save_dir, "predictive.backward.torch"))
    torch.save(losses_b, os.path.join(save_dir, "losses_b.torch"))
    torch.save(metrics_b, os.path.join(save_dir, "metrics.backward.torch"))

    return res_f, res_b


class ExpResultsWithLosses(ExperimentResults):
    def __init__(self, save_dir: str) -> None:
        super().__init__(save_dir)
        self.losses_f = torch.load(os.path.join(save_dir, "losses_f.torch"))
        self.losses_b = torch.load(os.path.join(save_dir, "losses_b.torch"))


class ExpResultsWithTwoLosses(ExpResultsWithLosses):
    def __init__(self, save_dir: str) -> None:
        super().__init__(save_dir)
        self.metrics_f = torch.load(os.path.join(save_dir, "metrics.forward.torch"))
        self.metrics_b = torch.load(os.path.join(save_dir, "metrics.backward.torch"))


def get_save_dir(save_dir_prefix: str, run: int) -> str:
    return os.path.join(save_dir_prefix, f"run={run:05}/")


def train_fb_n_times(train_d: BayesTrainData,
                     save_dir_prefix: str,
                     num_runs: int = 200,
                     dir_exists_silent: bool = False,
                     **kwargs) -> None:
    for run in tqdm.trange(num_runs):
        save_dir = get_save_dir(save_dir_prefix, run)
        if not os.path.isdir(save_dir):
            posterior_predictive_forward_and_backward(train_d, save_dir, use_tqdm=False, **kwargs)
        elif dir_exists_silent:
            print(f"Directory '{save_dir}' exists, won't overwrite")


def load_learning_curves(save_dir_prefix: str,
                         num_runs: int,
                         alt_metric: Optional[str] = None) -> LossDistribution:
    losses_f = []
    losses_b = []
    for run in range(num_runs):
        save_dir = get_save_dir(save_dir_prefix, run)
        if alt_metric is None:
            losses_f.append(torch.load(os.path.join(save_dir, "losses_f.torch")).tolist())
            losses_b.append(torch.load(os.path.join(save_dir, "losses_b.torch")).tolist())
        else:
            metrics_f = torch.load(os.path.join(save_dir, "metrics.forward.torch"))
            metrics_f = [dikt[alt_metric] for dikt in metrics_f]
            losses_f.append(metrics_f)

            metrics_b = torch.load(os.path.join(save_dir, "metrics.backward.torch"))
            metrics_b = [dikt[alt_metric] for dikt in metrics_b]
            losses_b.append(metrics_b)

    loss_dict = {"forward": losses_f, "backward": losses_b}
    with tempfile.NamedTemporaryFile() as file:
        write_json_to_file(loss_dict, file.name)
        result = LossDistribution(file.name)

    result.label = "ELBO-loss" if alt_metric is None else alt_metric

    return result
