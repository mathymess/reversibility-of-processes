import os
from typing import Tuple

import torch

import pyro
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal

import tqdm
import tempfile

from bayesian import (save_train_data_to_drive,
                      BayesianThreeFCLayers,
                      BayesTrainData,
                      ExperimentResults)
from train_test_utils import write_json_to_file, LossDistribution


def train_varinf(windows: torch.Tensor,
                 targets: torch.Tensor,
                 num_samples: int = 500,
                 hidden_size: int = 10,
                 num_epochs: int = 1000,
                 lr: float = 0.1,
                 use_tqdm: bool = True) -> Tuple[Predictive, torch.Tensor]:
    assert windows.ndim == targets.ndim == 2
    assert windows.shape[0] == targets.shape[0]

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    model = BayesianThreeFCLayers(window_len=windows.shape[-1], target_len=1,
                                  datapoint_size=1, hidden_size=hidden_size)
    guide = AutoDiagonalNormal(model)
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

    losses = []
    for i in tqdm.trange(num_epochs) if use_tqdm else range(num_epochs):
        loss = svi.step(windows, targets)
        losses.append(loss)

    predictive = Predictive(model=model, guide=guide, num_samples=num_samples)

    pyro.clear_param_store()  # Not sure this is necessary, being cautious.

    return predictive, torch.tensor(losses, dtype=torch.float32)


def posterior_predictive_forward_and_backward_impl(
        windows_f: torch.Tensor,
        targets_f: torch.Tensor,
        windows_b: torch.Tensor,
        targets_b: torch.Tensor,
        **kwargs) -> Tuple[Predictive, Predictive, torch.Tensor, torch.Tensor]:
    predictive_f, losses_f = train_varinf(windows_f, targets_f, **kwargs)
    predictive_b, losses_b = train_varinf(windows_b, targets_b, **kwargs)
    return predictive_f, predictive_b, losses_f, losses_b


def posterior_predictive_forward_and_backward(
        train_d: BayesTrainData,
        save_dir: str,
        **kwargs) -> Tuple[Predictive, Predictive, torch.Tensor, torch.Tensor]:
    save_train_data_to_drive(train_d, save_dir)
    predictive_f, predictive_b, losses_f, losses_b = posterior_predictive_forward_and_backward_impl(
        windows_f=train_d.windows_f,
        targets_f=train_d.targets_f,
        windows_b=train_d.windows_b,
        targets_b=train_d.targets_b,
        **kwargs)

    torch.save(predictive_f, os.path.join(save_dir, "predictive.forward.torch"))
    torch.save(predictive_b, os.path.join(save_dir, "predictive.backward.torch"))

    torch.save(losses_f, os.path.join(save_dir, "losses_f.torch"))
    torch.save(losses_b, os.path.join(save_dir, "losses_b.torch"))

    return predictive_f, predictive_b, losses_f, losses_b


class ExpResultsWithLosses(ExperimentResults):
    def __init__(self, save_dir: str) -> None:
        super().__init__(save_dir)

        self.losses_f = torch.load(os.path.join(save_dir, "losses_f.torch"))
        self.losses_b = torch.load(os.path.join(save_dir, "losses_b.torch"))


def get_save_dir(save_dir_prefix: str, run: int) -> str:
    return os.path.join(save_dir_prefix, f"run={run:05}/")


def train_fb_n_times(train_d: BayesTrainData,
                     save_dir_prefix: str,
                     num_runs: int = 200,
                     **kwargs) -> None:
    for run in tqdm.trange(num_runs):
        save_dir = get_save_dir(save_dir_prefix, run)
        if not os.path.isdir(save_dir):
            posterior_predictive_forward_and_backward(train_d, save_dir, use_tqdm=False, **kwargs)
        else:
            print(f"Directory '{save_dir}' exists, won't overwrite")


def load_learning_curves(save_dir_prefix: str, num_runs: int) -> LossDistribution:
    losses_f = []
    losses_b = []
    for run in range(num_runs):
        save_dir = get_save_dir(save_dir_prefix, run)
        losses_f.append(torch.load(os.path.join(save_dir, "losses_f.torch")).tolist())
        losses_b.append(torch.load(os.path.join(save_dir, "losses_b.torch")).tolist())

    loss_dict = {"forward": losses_f, "backward": losses_b}
    with tempfile.NamedTemporaryFile() as file:
        write_json_to_file(loss_dict, file.name)
        result = LossDistribution(file.name)

    return result
