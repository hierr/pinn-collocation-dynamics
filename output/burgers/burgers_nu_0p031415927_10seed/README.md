# Burgers ν = π/100: frozen ten-seed evaluation

This five-method evaluation uses paired seeds 4000–4009 and frozen **σ =
1e-4**, selected on separate development seeds 3000–3002. The [tuning
checkpoint](https://github.com/hierr/pinn-collocation-dynamics/commit/e089cb9d02def773fa16b9811e2f2f6ee5c59a7c)
contains all 15 initial-grid runs and selected the lowest-mean positive
candidate; its winner was interior, so no expansion was needed. Sigma zero was
a comparison baseline only, although it had lower mean solution error than all
positive candidates. See the [selection results](../../../docs/results.md#diffusion-selection-at-pi100)
and the [canonical tuning README](https://github.com/hierr/pinn-collocation-dynamics/blob/e089cb9d02def773fa16b9811e2f2f6ee5c59a7c/output/burgers/nu_0p031415927/sigma_0/README.md).
The tuning runs used source revision
[`cb6c6e8`](https://github.com/hierr/pinn-collocation-dynamics/commit/cb6c6e85642b82a70ce9e64013b74cd5633788ff).

## Design and reference

The [frozen configuration](../../../config/burgers_nu_0p031415927_10seed.yaml)
compares fixed Hammersley, upstream random resampling, RAD, PACMANN, and
PACMANN with diffusion. All methods retain the shared network, point budgets,
fixed boundary and initial constraints, optimizer schedule, and evaluation
grids. Only viscosity, reference dataset, and the development-selected sigma
differ from the preceding frozen comparison. Evaluation seeds are disjoint
from tuning seeds.

The N1024 reference
[data/burgers_dataset/burgers_nu_0p031415927_N1024.npz](../../../data/burgers_dataset/burgers_nu_0p031415927_N1024.npz)
was restored unchanged from dataset commit `52b9359`. Its stored viscosity is
**π/100 = 0.031415926535897934**, and its SHA-256 is
`aad9aff8eecc32e1fe9e60ad2e4ca5f399dd5e96bf748851f085a2165923fb60`.
Structural, constraint, and loader checks passed, but the reference has no
independent numerical-accuracy certification.

Use final-model solution and independent residual errors on all ten seeds.
Retain every planned outcome. Report per-seed values, mean and sample SD,
paired differences and improvement counts. Supplementary trimmed means remove
one minimum and one maximum independently for solution L2 and residual RMS.
Solution relative L2 is reported as percent; paired solution differences are
percentage points; residual RMS retains its original units. Boundary and
initial-condition RMSE are separate diagnostics. Residual error uses the same
selected sigma; do not retune on evaluation seeds.

## Reproduction

Use the [documented environment](../../../README.md#setup-and-execution).
The runner selects this YAML and methods in the order listed above. From the
repository root, execute:

```sh
venv/bin/python -u -m scripts.run_experiments.run_burgers_campaign
```

The runner launches ten concurrent seeds per method, with one numerical CPU
thread per worker, on one NVIDIA GeForce RTX 5070 (12 GB) using float64.
Methods run sequentially; each batch completes training and evaluation before
the next begins. Training time includes compilation and optimizer work and
excludes final evaluation. It reflects ten-way GPU concurrency and is not
comparable to three-way tuning timings.

Runs are retained under
`output/burgers/burgers_nu_0p031415927_10seed/<method>/seed_<seed>/`.
Consolidated comparisons are in [results](../../../docs/results.md);
see the [study context](../../../docs/research_context.md) and
[artifact dictionary](../../../docs/output_format.md). Per-seed outcomes and
aggregates are added here after the planned evaluation completes.
