import os
from typing import Tuple

import torch

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


def get_rmse_loss(windows: torch.Tensor,
                  targets: torch.Tensor,
                  model: pyro.nn.PyroModule,
                  guide: pyro.infer.autoguide.AutoGuide,
                  num_samples: int = 200) -> float:
    predictive = Predictive(model=model, guide=guide, num_samples=num_samples)
    preds = predictive(windows)["obs"]
    return quality_metrics(preds, targets)["rmse"]


TrainRetval = Tuple[Predictive, torch.Tensor, torch.Tensor]


def train_varinf(windows: torch.Tensor,
                 targets: torch.Tensor,
                 num_samples: int = 500,
                 hidden_size: int = 10,
                 num_epochs: int = 1000,
                 lr: float = 0.1,
                 use_tqdm: bool = True) -> TrainRetval:
    assert windows.ndim == targets.ndim == 2
    assert windows.shape[0] == targets.shape[0]

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    model = BayesianThreeFCLayers(window_len=windows.shape[-1], target_len=1,
                                  datapoint_size=1, hidden_size=hidden_size)
    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

    losses = []
    rmse_losses = []
    for i in tqdm.trange(num_epochs) if use_tqdm else range(num_epochs):
        loss = svi.step(windows, targets)
        losses.append(loss)

        if i % 10 == 0:
            rmse = get_rmse_loss(windows, targets, model, guide)
            rmse_losses.append(rmse)

    losses = torch.tensor(losses, dtype=torch.float32)
    rmse_losses = torch.tensor(rmse_losses, dtype=torch.float32)

    predictive = Predictive(model=model, guide=guide, num_samples=num_samples)

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    return predictive, losses, rmse_losses


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

    predictive_f, losses_f, rmse_losses_f = res_f
    torch.save(predictive_f, os.path.join(save_dir, "predictive.forward.torch"))
    torch.save(losses_f, os.path.join(save_dir, "losses_f.torch"))
    torch.save(rmse_losses_f, os.path.join(save_dir, "rmse_losses_f.torch"))

    predictive_b, losses_b, rmse_losses_b = res_b
    torch.save(predictive_b, os.path.join(save_dir, "predictive.backward.torch"))
    torch.save(losses_b, os.path.join(save_dir, "losses_b.torch"))
    torch.save(rmse_losses_b, os.path.join(save_dir, "rmse_losses_b.torch"))

    return res_f, res_b


class ExpResultsWithLosses(ExperimentResults):
    def __init__(self, save_dir: str) -> None:
        super().__init__(save_dir)
        self.losses_f = torch.load(os.path.join(save_dir, "losses_f.torch"))
        self.losses_b = torch.load(os.path.join(save_dir, "losses_b.torch"))


class ExpResultsWithTwoLosses(ExpResultsWithLosses):
    def __init__(self, save_dir: str) -> None:
        super().__init__(save_dir)
        self.rmse_losses_f = torch.load(os.path.join(save_dir, "rmse_losses_f.torch"))
        self.rmse_losses_b = torch.load(os.path.join(save_dir, "rmse_losses_b.torch"))


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
                         rmse_instead: bool = False) -> LossDistribution:
    losses_f = []
    losses_b = []
    for run in range(num_runs):
        save_dir = get_save_dir(save_dir_prefix, run)
        rmse_pref = "rmse_" if rmse_instead else ""
        losses_f.append(torch.load(os.path.join(save_dir, rmse_pref + "losses_f.torch")).tolist())
        losses_b.append(torch.load(os.path.join(save_dir, rmse_pref + "losses_b.torch")).tolist())

    loss_dict = {"forward": losses_f, "backward": losses_b}
    with tempfile.NamedTemporaryFile() as file:
        write_json_to_file(loss_dict, file.name)
        result = LossDistribution(file.name)

    return result
