"""
Experiment sweep for β-DP-SGD.

Matches the paper's hyperparameters exactly:
  β ∈ 8 values in [1.0, 2.0]
  σ ∈ 6 evenly spaced in [0.5, 3.0]
  batch_size ∈ {128, 256}
  LR ∈ {0.5, 1.0}
  max_grad_norm ∈ {0.05, 0.1, 0.25, 0.5}
  δ = 10^{-6}
  epochs = 100 (300 for LSTM)
  trials = 3

Usage:
  python run_sweep.py --job-index <SLURM_ARRAY_TASK_ID> --output-dir <path>
  python run_sweep.py --list-jobs          # print total number of jobs
  python run_sweep.py --describe-job 42    # print config for job 42
"""

import argparse
import itertools
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.simplefilter("ignore")

# ---------------------------------------------------------------------------
# Hyperparameter grid (matches paper appendix)
# ---------------------------------------------------------------------------
BETAS = [round(b, 4) for b in np.linspace(1.0, 2.0, 8)]
SIGMAS = [round(s, 2) for s in np.linspace(0.5, 3.0, 6)]
BATCH_SIZES = [128, 256]
LRS = [0.5, 1.0]
MAX_GRAD_NORMS = [0.05, 0.1, 0.25, 0.5]
DELTA = 1e-6
EPOCHS = 100
LSTM_EPOCHS = 300
TRIALS = 3
MAX_PHYSICAL_BATCH_SIZE = 1024
MOMENTUM = 0.9

DATASET_MODEL_MAP = {
    "cifar-10": "scatternet_cnns",
    "svhn": "scatternet_cnns",
    "adult": "adult_FCN",
    "imdb": "LSTM",
}

DATASETS = list(DATASET_MODEL_MAP.keys())


def build_job_grid():
    """Return list of all (dataset, beta, sigma, batch_size, LR, max_grad_norm) configs."""
    return list(
        itertools.product(DATASETS, BETAS, SIGMAS, BATCH_SIZES, LRS,
                          MAX_GRAD_NORMS))


def config_from_index(index):
    grid = build_job_grid()
    if index < 0 or index >= len(grid):
        raise ValueError(
            f"Job index {index} out of range [0, {len(grid) - 1}]")
    dataset, beta, sigma, batch_size, lr, max_grad_norm = grid[index]
    model_name = DATASET_MODEL_MAP[dataset]
    epochs = LSTM_EPOCHS if model_name == "LSTM" else EPOCHS
    return {
        "dataset": dataset,
        "model_name": model_name,
        "beta": beta,
        "sigma": sigma,
        "batch_size": batch_size,
        "LR": lr,
        "max_grad_norm": max_grad_norm,
        "delta": DELTA,
        "epochs": epochs,
        "trials": TRIALS,
    }


# ---------------------------------------------------------------------------
# Import experiment machinery from existing code
# ---------------------------------------------------------------------------

def _setup_imports():
    """Add experiment dir to path so existing modules are importable."""
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    if exp_dir not in sys.path:
        sys.path.insert(0, exp_dir)
    # Also need project root for data_aware_dp
    project_root = os.path.abspath(os.path.join(exp_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def run_single_config(cfg, output_dir, job_index):
    """Run all trials for a single hyperparameter config."""
    _setup_imports()

    from beta_image_classifier import (
        train,
        test,
        initialize_model_scatternet_cnns,
        intialize_model_adult_CNN,
        intialize_model_LSTM,
    )
    import dataset_management
    from data_aware_dp import sampling
    from opacus import PrivacyEngine
    from opacus.accountants.analysis.prv import prvs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = cfg["dataset"]
    model_name = cfg["model_name"]
    beta = cfg["beta"]
    sigma = cfg["sigma"]
    batch_size = cfg["batch_size"]
    lr = cfg["LR"]
    max_grad_norm = cfg["max_grad_norm"]
    delta = cfg["delta"]
    epochs = cfg["epochs"]
    trials = cfg["trials"]

    scale = float(prvs.sigma_to_scale(sigma))

    # ---- data loaders ----
    data_root_dir = "".join(dataset.split("-")).upper()
    DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             data_root_dir)

    if dataset == "cifar-10":
        train_loader, test_loader = dataset_management.initialize_data_CIFAR10(
            batch_size=batch_size, DATA_ROOT=DATA_ROOT)
    elif dataset == "svhn":
        train_loader, test_loader = dataset_management.initialize_data_SVHN(
            batch_size=batch_size, DATA_ROOT=DATA_ROOT)
    elif dataset == "adult":
        train_loader, test_loader = dataset_management.initialize_data_ADULT(
            batch_size=batch_size, DATA_ROOT=DATA_ROOT)
    elif dataset == "imdb":
        train_loader, test_loader = dataset_management.initialize_data_IMDB(
            batch_size=batch_size, DATA_ROOT=DATA_ROOT)

    # ---- model initializer ----
    model_initializers = {
        "scatternet_cnns": initialize_model_scatternet_cnns,
        "adult_FCN": intialize_model_adult_CNN,
        "LSTM": intialize_model_LSTM,
    }
    model_init_fn = model_initializers[model_name]

    # ---- run trials ----
    all_results = []

    for trial in range(trials):
        logging.info(
            f"Trial {trial+1}/{trials} | {dataset}/{model_name} | "
            f"β={beta} σ={sigma} bs={batch_size} lr={lr} C={max_grad_norm}")
        try:
            model, criterion, optimizer, privacy_engine_ = model_init_fn(
                device=device,
                train_loader=train_loader,
                bn_noise_multiplier=8,
                LR=lr,
                momentum=MOMENTUM)

            beta_sampler = sampling.beta_exponential_sampler__torch(
                beta=beta,
                scale=scale * max_grad_norm,
                device=device)

            model, optimizer, train_loader_priv = privacy_engine_.make_private(
                noise_multiplier=sigma,
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                max_grad_norm=max_grad_norm,
                beta=beta,
                beta_sampler=beta_sampler)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=10,
                                                         gamma=1.0)

            accs, test_accs, losses, epsilons = [], [], [], []
            best_test_acc = 0.0
            patience_counter = 0

            for epoch in range(epochs):
                top1_acc, loss, epsilon = train(model, train_loader_priv,
                                                optimizer, epoch + 1, device,
                                                privacy_engine_, delta, beta)
                scheduler.step()

                test_acc = float(test(model, test_loader, device))

                accs.append(float(top1_acc))
                losses.append(float(loss))
                epsilons.append(float(epsilon))
                test_accs.append(test_acc)

                # early stopping: 10 epochs without improvement
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 10:
                    logging.info(
                        f"  Early stop at epoch {epoch+1} (no improvement for 10 epochs)"
                    )
                    break

                # stop if privacy budget blown
                if epsilon > 10.0:
                    logging.info(
                        f"  Privacy budget exceeded at epoch {epoch+1} (ε={epsilon:.2f})"
                    )
                    break

            trial_result = {
                "trial": trial,
                "train_acc": accs,
                "test_acc": test_accs,
                "train_loss": losses,
                "epsilons": epsilons,
                "best_test_acc": best_test_acc,
            }
            all_results.append(trial_result)

        except Exception as e:
            logging.error(f"  Trial {trial+1} failed: {e}")
            all_results.append({"trial": trial, "error": str(e)})
            continue

    # ---- save results ----
    result = {
        "config": cfg,
        "job_index": job_index,
        "trials": all_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    fname = (f"job_{job_index:05d}__{dataset}__{model_name}__beta_{beta}"
             f"__sigma_{sigma}__bs_{batch_size}__lr_{lr}"
             f"__C_{max_grad_norm}.yaml")
    save_path = os.path.join(output_dir, fname)

    with open(save_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False)

    logging.info(f"Results saved to {save_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="β-DP-SGD experiment sweep")
    parser.add_argument("--job-index",
                        type=int,
                        default=None,
                        help="SLURM_ARRAY_TASK_ID — index into the job grid")
    parser.add_argument("--output-dir",
                        type=str,
                        default="results",
                        help="Directory to save result YAMLs")
    parser.add_argument("--list-jobs",
                        action="store_true",
                        help="Print total number of jobs and exit")
    parser.add_argument("--describe-job",
                        type=int,
                        default=None,
                        help="Print config for a given job index and exit")
    parser.add_argument("--data-dir",
                        type=str,
                        default=None,
                        help="Override root data directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    grid = build_job_grid()
    total_jobs = len(grid)

    if args.list_jobs:
        print(f"Total jobs: {total_jobs}")
        print(f"  Datasets: {DATASETS}")
        print(f"  Betas ({len(BETAS)}): {BETAS}")
        print(f"  Sigmas ({len(SIGMAS)}): {SIGMAS}")
        print(f"  Batch sizes: {BATCH_SIZES}")
        print(f"  Learning rates: {LRS}")
        print(f"  Max grad norms: {MAX_GRAD_NORMS}")
        print(f"  Trials per job: {TRIALS}")
        print(f"  Total training runs: {total_jobs * TRIALS}")
        return

    if args.describe_job is not None:
        cfg = config_from_index(args.describe_job)
        print(json.dumps(cfg, indent=2))
        return

    if args.job_index is None:
        parser.error("--job-index is required (or use --list-jobs)")

    cfg = config_from_index(args.job_index)
    logging.info(f"Job {args.job_index}/{total_jobs - 1}: {json.dumps(cfg)}")

    run_single_config(cfg, args.output_dir, args.job_index)


if __name__ == "__main__":
    main()
