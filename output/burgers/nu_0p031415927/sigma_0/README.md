# Burgers diffusion tuning at ν = π/100

This completed development sweep selected the positive **σ = 1e-4** using
paired seeds 3000–3002. The winner is interior among the four tested positive
settings, so no expansion was required. Ordinary PACMANN at σ = 0 was the
comparison baseline only and had lower mean solution error than every positive
candidate; the approved rule nevertheless selects only among positive values.

## Reference and design

The tracked N1024 reference
[data/burgers_dataset/burgers_nu_0p031415927_N1024.npz](../../../../data/burgers_dataset/burgers_nu_0p031415927_N1024.npz)
was restored unchanged from dataset commit [52b9359](https://github.com/hierr/pinn-collocation-dynamics/commit/52b9359).
Its stored viscosity is **π/100 = 0.031415926535897934**; the filename is
rounded. Its SHA-256 is
`aad9aff8eecc32e1fe9e60ad2e4ca5f399dd5e96bf748851f085a2165923fb60`.
The reference passed structural and production-loader checks, but has no
independent numerical-accuracy certification.

The [five candidate configurations](../../../../config/burgers/nu_0p031415927/)
compare ordinary PACMANN with diffusion coefficients 1e-5, 1e-4, 1e-3, and
1e-2. Every candidate uses the same network, point budgets, training schedule,
evaluation grids, and seeds. Each has three paired development seeds; three
seed processes run concurrently per candidate, and candidates run sequentially.
The positive setting with the lowest arithmetic mean final solution relative
L2 is selected; exact ties favor the smaller positive sigma. Sigma zero cannot
win selection. Residual error is reported at the same solution-selected sigma
without a second selection. A boundary winner would trigger the approved
two-decade expansion, with at most two rounds; this sweep's winner was interior
and required no expansion.

From the repository root, using the
[documented environment](../../../../README.md#setup-and-execution), run:

```sh
venv/bin/python -m scripts.run_experiments.run_burgers_sigma_tuning
```

The existing runner selects all five YAMLs in ascending order. Runs are retained
under `output/burgers/nu_0p031415927/sigma_<sigma>/<method>/seed_<seed>/`.
Training metadata records source revision
[`cb6c6e8`](https://github.com/hierr/pinn-collocation-dynamics/commit/cb6c6e85642b82a70ce9e64013b74cd5633788ff),
the shared pinned environment, CUDA device, and reference fingerprint. Runs
used float64 on one NVIDIA GeForce RTX 5070 (12 GB), with one numerical CPU
thread per worker.

All 15 planned runs completed without numerical failures. Required training,
model, point, prediction, metric, and diagnostic artifacts are present for each
run. Initial interior designs and constraints match exactly across candidates
within each seed, constraints remained fixed, and physics points did not move
during L-BFGS. An independent CPU evaluation reloaded all 15 final checkpoints
and reproduced solution MSE, solution relative L2, residual RMS, boundary RMSE,
and initial RMSE within 1e-12 + 1e-10·|saved|.

## Solution outcomes

Solution relative L2 is reported as percent. The mean and sample SD use all
three seeds. Paired changes are candidate minus the same-seed sigma-zero
baseline, in percentage points; negative values favor diffusion.

| Sigma | Seed 3000 | Seed 3001 | Seed 3002 | Mean ± sample SD (%) | Paired change vs. zero (pp) | Improved / 3 | Training seconds per run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00997495 | 0.0218088 | 0.00886318 | 0.0135490 ± 0.00717480 | — | — | 3239.2 ± 4.4 |
| 1e-5 | 0.0160298 | 0.0212499 | 0.00988388 | 0.0157212 ± 0.00568930 | +0.00217220 | 1/3 | 3165.9 ± 36.5 |
| **1e-4** | **0.0112303** | **0.0186738** | **0.0128486** | **0.0142509 ± 0.00391486** | **+0.000701916** | **1/3** | **3296.5 ± 21.4** |
| 1e-3 | 0.0158911 | 0.0151572 | 0.0198399 | 0.0169627 ± 0.00251858 | +0.00341373 | 1/3 | 3252.4 ± 1.0 |
| 1e-2 | 0.0261598 | 0.0275000 | 0.0446181 | 0.0327593 ± 0.0102919 | +0.0192103 | 0/3 | 3163.9 ± 44.7 |

The positive-candidate mean is lowest at sigma 1e-4. The sigma-zero baseline
mean remains lower by 0.000701916 percentage points; that baseline is not an
eligible setting under the accepted positive-only selection rule.

## Residual outcomes

Residual RMS is reported in its original unnormalized units. Residuals do not
select sigma.

| Sigma | Seed 3000 | Seed 3001 | Seed 3002 | Mean ± sample SD |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000676362 | 0.00105693 | 0.00114483 | 0.000959376 ± 0.000249007 |
| 1e-5 | 0.000860667 | 0.00123847 | 0.000988823 | 0.00102932 ± 0.000192131 |
| **1e-4** | **0.000842913** | **0.00105939** | **0.00100189** | **0.000968064 ± 0.000112133** |
| 1e-3 | 0.00109919 | 0.000975587 | 0.00130863 | 0.00112780 ± 0.000168357 |
| 1e-2 | 0.00189462 | 0.00186518 | 0.00243286 | 0.00206422 ± 0.000319589 |

At sigma 1e-4, the mean paired residual difference from sigma zero is
+0.00000868754 ± 0.000154842, with lower residual in one of three seeds.
The small development sample and a single viscosity do not establish a general
diffusion benefit or a sharpness effect.

## Observed optimizer work

Each run performed 35,000 Adam steps. The table gives mean ± sample SD of
observed L-BFGS iterations and function evaluations per run. Training seconds
include compilation and training and exclude post-processing; they reflect
three-way GPU concurrency.

| Sigma | L-BFGS iterations | L-BFGS function evaluations |
| ---: | ---: | ---: |
| 0 | 16891 ± 656.2 | 27411.7 ± 2907.3 |
| 1e-5 | 16463 ± 482.6 | 26983.7 ± 1371.1 |
| **1e-4** | **17101.7 ± 486.1** | **28792 ± 2102.0** |
| 1e-3 | 16828 ± 581.7 | 27838.3 ± 1712.9 |
| 1e-2 | 16368 ± 83.1 | 27014 ± 2628.0 |

These observed counts and all per-seed records are retained in each run's
`training.json` and `metrics.json`. The summary comparison is in
[docs/results.md](../../../../docs/results.md#diffusion-tuning-at-pi100);
the [study context](../../../../docs/research_context.md) describes the shared
selection policy, and the [artifact dictionary](../../../../docs/output_format.md) defines retained files. A separate ten-seed evaluation uses the frozen positive
sigma on new seeds after the setting is transferred to `main`.
