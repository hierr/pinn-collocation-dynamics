# Experimental notes

Only implementation validation has been run; no scientific reproduction campaign
has been completed.

For each material experiment, add its objective/hypothesis, controlled and changed
variables, metrics, reproducibility requirements, and decision criteria before
execution. Afterwards record results, limitations, conclusion, and next steps.

## 2026-10-06: Two-method implementation validation

Objective: validate the minimal released-code Burgers reproduction before any
accuracy campaign. This is an engineering smoke check, not a test of sampler
superiority or a reproduction of the paper's error statistics.

Controls: pinned released dependencies with the PyTorch CUDA 12.8 wheel,
float64, four 64-neuron tanh layers, original dataset and checksum, seed 1000,
2,500 interior / 80 boundary / 160 initial points, and common Hammersley design.
Changed variable: fixed Hammersley versus PACMANN-Adam. The validation budget
is two phases of 100 Adam steps followed by L-BFGS configured for 20 iterations,
with history display/evaluation every 50 steps. PACMANN retains period 50,
15 steps, step size 1e-5, betas 0.9/0.999, and epsilon 1e-7.

Run both methods on the host GPU; preserve each attempt under a fresh campaign.
Require four PACMANN events, fixed constraint coordinates, unchanged physics
coordinates in L-BFGS, unchanged Hammersley coordinates, finite metrics,
complete artifacts, and weight-reload agreement (rtol 1e-10, atol 1e-12).
Record actual L-BFGS endpoints; early convergence is acceptable. Preserve the
upstream initial PDE population of 2,740 points and its reduction to 2,500 at
PACMANN's first event. This is a disclosed difference from the static baseline.

Artifacts remain in ignored local output directories; no scientific campaign
runs until a durable archive policy is agreed. Tests compare the coordinate
callback against the actual pinned upstream class, check derivatives/data
ordering, and verify that run directories cannot be overwritten.

Results and decision: both methods completed in `burgers_validation_03` on an
NVIDIA GeForce RTX 5050 Laptop GPU with CUDA execution. All constraint and
movement invariants passed. PACMANN recorded exactly four events; both methods
reloaded final weights with zero prediction difference. Final MSE / relative L2
were 0.0918415 / 0.493302 for fixed Hammersley and 0.0977273 / 0.508863 for
PACMANN. These deliberately short runs remain far from an accuracy campaign;
these numbers do not support a method comparison.

The first attempts (`burgers_validation_01`, both methods) failed the reload
check because a fresh DeepXDE model needs compilation before prediction. Those
attempts also exposed import-time L-BFGS per-call limits of 1,000 despite a
configured limit of 20. The fixed-method attempt in `burgers_validation_02`
confirmed the limit correction but still failed reload. All failed directories
are retained. Initial attempts overlapped on the GPU; their timings are unusable.
The final successful pair ran sequentially.

DeepXDE checks its total L-BFGS iteration limit between optimizer calls. A call
that ends at its function-evaluation limit can be followed by another full call,
overshooting the configured iteration count. Final actual L-BFGS counts were
20 and 38 for fixed Hammersley, and 20 and 20 for PACMANN. These endpoints are
archived, not silently treated as equal budgets. The runner preserves this
released-library behavior; define a strict stopping policy before a matched-budget
study, and disclose any change from upstream.

Four focused tests pass. Unit-test coverage alone is 53%; combined coverage
with instrumented GPU validation is 92%. Ruff, ty, lock consistency, and installed
dependency checks pass. Upstream files and dataset remain unchanged. Next:
review this implementation checkpoint, establish campaign artifact storage and
numerical reproduction acceptance criteria, then authorize full reproduction.
Diffusion is not implemented in this checkpoint.
