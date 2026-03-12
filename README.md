# The Picard-Lagrange Framework for Higher-Order Langevin Monte Carlo

This repository contains the code for the paper **“The Picard-Lagrange Framework for Higher-Order Langevin Monte Carlo.”** (arXiv:2510.18242)

**Links:** [arXiv](https://arxiv.org/abs/2510.18242) · [Code](https://github.com/kaihongz/HigherOrderLMC)

## Overview

The repository provides:

- implementation of the higher-order Langevin sampler proposed in the paper;
- baseline implementations of overdamped and underdamped Langevin Monte Carlo;
- scripts for the Gaussian-target experiments;
- scripts for the Bayesian logistic regression (BLR) experiments on simulated and real data.

The code is written in Python and runs on CPU with NumPy/SciPy. No GPU is required.

## Repository structure

```text
.
├── higher_order_langevin_fast.py   # higher-order Langevin sampler (K >= 3)
├── baseline.py                     # OD-LMC and UD-LMC baselines
├── blr_utils.py                    # shared utilities for BLR experiments
├── experiment.py                   # Gaussian target comparison
├── experiment_order_sweeps.py      # h / gamma sweeps for higher-order samplers
├── experiment_blr_simulated.py     # simulated BLR experiment
└── experiment_blr_real.py          # real-data BLR experiment
```

## Requirements

The code depends on the following Python packages:

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`


## Reproducing the experiments

Each experiment script contains a small configuration block near the top of the file. To reproduce the figures in the paper, adjust those parameters if needed and run the corresponding script from the repository root.

### 1. Gaussian targets

This experiment compares OD-LMC, UD-LMC, HO-LMC with `K=3`, and HO-LMC with `K=4` on diagonal Gaussian targets.

```bash
python experiment.py
```

By default, this script writes its outputs to `./gaussian_mc_clean/`.
### 2. Higher-order parameter sweeps on Gaussian targets

This script produces the `h`-sweep and `gamma`-sweep plots for the higher-order samplers.

```bash
python experiment_order_sweeps.py
```

Outputs are written to `./gaussian_mc_order_sweeps/`.

### 3. Simulated Bayesian logistic regression

This experiment evaluates the samplers on simulated Bayesian logistic regression problems.

```bash
python experiment_blr_simulated.py
```

Outputs are written to a timestamped directory under `./blr_clean/simulated_outputs/`.

### 4. Real-data Bayesian logistic regression

This experiment evaluates the samplers on real binary-classification data sets.

```bash
python experiment_blr_real.py
```

Outputs are written to a timestamped directory under `./blr_clean/real_outputs/`.

By default, this script saves:

- one log-loss figure;
- one AUROC figure;
- one accuracy figure;

## Using the sampler in your own code

A minimal example is:

```python
import numpy as np
from higher_order_langevin_fast import HigherOrderLangevin


def grad_U(x: np.ndarray) -> np.ndarray:
    return x  # example: U(x) = 0.5 * ||x||^2

sampler = HigherOrderLangevin(
    K=3,
    d=10,
    h=0.05,
    gamma=4.0,
    grad_U_fn=grad_U,
    nu_star=2,
)

samples = sampler.sample(N_steps=1000, burn_in=200)
print(samples.shape)
```

Baseline samplers are available in `baseline.py`:

- `OverdampedLMC`
- `UnderdampedLangevinExp`

## Citation

If you use this code in your research, please cite the paper.

```bibtex
@article{mahajan2025picard,
  title={The Picard-Lagrange Framework for Higher-Order Langevin Monte Carlo},
  author={Mahajan, Jaideep and Zhang, Kaihong and Liang, Feng and Liu, Jingbo},
  journal={arXiv preprint arXiv:2510.18242},
  year={2025}
}
```

