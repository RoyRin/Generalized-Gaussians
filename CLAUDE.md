# Project Context

Research project for the paper "Beyond Laplace and Gaussian: Exploring the Generalized Gaussian Mechanism for Private Machine Learning". Branch `rr/experiments`.

## What the research does

The Generalized Gaussian (GG) mechanism parameterized by β generalizes Laplace (β=1) and Gaussian (β=2) noise for differential privacy. We integrate this into DP-SGD (called β-DP-SGD) by replacing Gaussian noise with GG noise and using ℓ_β norm clipping. The main finding is that β=2 (Gaussian) is nearly optimal.

## Codebase structure

- **`Opacus-PRV/`** — Forked Opacus library with PRV accountant extended for GG mechanisms
  - `opacus/accountants/prv.py` — PRV privacy accountant (handles arbitrary β)
  - `experiments/beta_image_classifier.py` — Original experiment script (train/test/model-init functions)
  - `experiments/run_sweep.py` — Config-driven experiment runner for the full hyperparameter sweep
  - `experiments/submit_sweep.sbatch` — SLURM array job submission script
  - `experiments/dataset_management.py` — Data loaders for CIFAR-10, SVHN, Adult, IMDB
  - `experiments/local_models.py` — Model definitions (CNN, FCN, LSTM)
  - `experiments/scatternet_cnns.py` — ScatterNet CNN models (used for CIFAR-10 and SVHN)
- **`data_aware_dp/`** — Core research code
  - `sampling.py` — GG samplers (numpy, torch, JAX)
  - `rdp.py` — RDP calculations for GG
  - `models.py` — ResNet9, FCN, etc.
- **`overleaf/`** — Git submodule with the paper's LaTeX source (from Overleaf)

## The experiment sweep (`run_sweep.py`)

This reruns the full β-DP-SGD experiments matching the paper's appendix hyperparameters:

| Parameter | Values |
|-----------|--------|
| β | 8 values evenly spaced in [1.0, 2.0] |
| σ (noise multiplier) | 6 values evenly spaced in [0.5, 3.0] |
| batch_size | {128, 256} |
| learning rate | {0.5, 1.0} |
| max_grad_norm (clip) | {0.05, 0.1, 0.25, 0.5} |
| δ | 10⁻⁶ |
| epochs | 100 (300 for LSTM) |
| trials | 3 per config |

**Datasets and models:**
- CIFAR-10 → ScatterNet CNN
- SVHN → ScatterNet CNN
- Adult → 2-layer FCN
- IMDB → LSTM (~1M params)

**Total: 3072 SLURM jobs**, each running 3 trials of one (dataset, β, σ, batch_size, LR, max_grad_norm) config. Results saved as individual YAML files in `results/`.

## How to run

```bash
cd Opacus-PRV/experiments
python run_sweep.py --list-jobs        # verify 3072 jobs
python run_sweep.py --describe-job 42  # inspect a specific config
mkdir -p logs results
sbatch submit_sweep.sbatch             # launch all jobs
```

Edit `submit_sweep.sbatch` to load your cluster's CUDA modules and Python environment. The `--array=0-3071%64` controls max concurrent jobs.

## Key motivation for rerun

The paper's appendix has a TODO about the batch-size ablation (batch_size=128) showing non-monotonic behavior on SVHN/CIFAR-10. We're rerunning the full suite with real results (not digitized from old figures) to resolve this and get clean data with proper std devs across all ablations (learning rate, clipping norm, batch size, noise multiplier).
