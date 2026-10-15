# Burgers N1024 sigma tuning

Scientific development: select sigma for the generated Burgers reference at
`nu = 0.01/pi`. Compare ordinary PACMANN (sigma zero) with four diffusion scales,
using three paired development seeds (`3000`, `3001`, `3002`): fifteen runs on
one remote CUDA machine with the locked environment. These seeds are excluded
from later frozen-setting evaluation. The reference is used only for evaluation.

## Settings and selection

| Sigma | Configuration |
| ---: | --- |
| 0 | [PACMANN](../../config/burgers_sigma_baseline.yaml) |
| 1e-5 | [Diffusion](../../config/burgers_sigma_1e-5.yaml) |
| 1e-4 | [Diffusion](../../config/burgers_sigma_1e-4.yaml) |
| 1e-3 | [Diffusion](../../config/burgers_sigma_1e-3.yaml) |
| 1e-2 | [Diffusion](../../config/burgers_sigma_1e-2.yaml) |

Match initial weights, point populations, constraints, network, optimizer
budgets, and evaluation grids across settings; only sigma varies. Evaluate final
checkpoints using ordinary solution relative L2 on the common uniform grid.
Select the lowest mean across all planned seeds; exact ties favor smaller sigma,
including zero. A positive winner at either search boundary requires review
before freezing; do not expand the search automatically. Report individual
errors, sample SD, and paired changes. Use the same selected sigma for residual
error; do not select a second residual-optimal setting. Preserve failures and
unfinished runs; do not select from only the successful subset.

## Optimizer work

DeepXDE 1.14's PyTorch L-BFGS driver checks `maxiter` between optimizer calls.
Some stages therefore exceeded the configured 3,000-iteration limit; the
observed range was 3,000–3,620. Settings are matched across methods, while actual
optimizer work varies. Each run's `training.json` records stage iterations,
function evaluations, and time.

## Remote execution

Pull `hyperparameter-tuning`, use the existing environment, and verify CUDA
execution before starting. From the repository root, with `ACTION = "run"`:

```bash
.venv/bin/python -m scripts.run_experiments.run_burgers_sigma_tuning
```

Set `ACTION = "summarize"` in that script and use the same command to print saved
results without training. The runner checks all destinations before executing
baseline then increasing sigma, evaluating and plotting each run after training.
Existing runs cannot be overwritten. This README holds the shared design; the
other setting directories link here.

[Study context](../../docs/research_context.md#accepted-controls-and-evaluation-policy)
· [Results](../../docs/results.md) · [Artifacts](../../docs/output_format.md)
· [Environment](../../README.md#setup-and-execution)
