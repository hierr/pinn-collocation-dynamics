# Burgers diffusion tuning at ν = 0.001

Prepared scientific development for a sharper regime; training has not started.
Evaluation-grid adequacy remains a prerequisite for sigma selection.

## Reference and controls

Use `data/burgers_dataset/burgers_nu_0p001_N4096.npz`, transferred unchanged
from `dataset-validation` checkpoint `41529cb`. Its SHA-256 is
`b628305506f4d3fcfc5052c725bfe595fee716bce21e951165a8ce0af85d89de`.
N4096 denotes Chebyshev degree, giving 4,097 spatial nodes. The solver uses
Radau with an analytical Jacobian, 201 snapshots through `t = 1`,
`rtol = 1e-11`, `atol = 1e-13`, and `max_step = 0.005`.

N2048–N4096 solution relative L2 disagreement is 0.00964855% on 8,193 uniform
points and 199 shared times through `t = 0.99`; spatial-gradient disagreement
is 1.00069%. This supports N4096 as the working reference, without an absolute
error bound or temporal-convergence certification. Resolution artifacts remain
on `dataset-validation`. The reference is used for evaluation only.

The [five configurations](../../../../config/burgers/nu_0p001/) retain the
shared network, optimizer schedule, point budgets, and paired seeds 3000–3002.
The initial grid compares ordinary PACMANN with positive sigma values
`1e-5`, `1e-4`, `1e-3`, and `1e-2`: 15 planned runs. Candidates run sequentially,
with three concurrent seeds per candidate and one numerical CPU thread per worker.
The first baseline seed runs alone for the evaluation-grid check; its timing
has different concurrency from the later tuning batches.

## Evaluation and selection

The configured 1,024-point solution grid and 1,024-by-200 residual midpoint grid
are provisional. Train only planned PACMANN baseline seed 3000 first. Using its
fixed saved weights and the same N4096 reference, compare solution relative L2
and residual RMS at 1,024 versus 4,096 spatial evaluation points. Keep the 199
native solution times through `0.99` and 200 residual midpoint times unchanged.
For each metric, compute `100 * abs(coarse - fine) / abs(fine)`. Retain the
current grid if both changes are below 1%. If either fails, the bounded fallback
compares 4,096 with 8,192, then 8,192 with 16,384
spatial points, retaining the first coarser grid passing both metrics. Stop if
none passes. This screens spatial sampling for one model and does not certify
temporal resolution. Revisit only if sensitivity appears or conclusions depend
on differences of comparable size.

Select the positive sigma with the lowest arithmetic mean final solution
relative L2 over all three seeds; exact ties favor the smaller sigma. Sigma zero
is a comparison baseline only. A positive boundary winner triggers expansion
by one and two decades on that side, with at most two expansion rounds. Freeze
the best tested positive value and disclose any remaining boundary limitation.
Preserve numerical failures for review. Report residual error at the same
solution-selected sigma. Ten-seed evaluation follows separately with frozen settings.

## Reproduction

From the repository root in the [documented environment](../../../../README.md#setup-and-execution):

Run only the first baseline seed:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python -m scripts.run_experiments.run_burgers_baseline
```

The baseline entry point archives the full three-seed configuration but executes
only seed 3000. Check its fixed weights and resolve the evaluation settings with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python -m scripts.post_processing.check_burgers_evaluation_grid
```

The checker writes `evaluation_grid_check.json` in this directory and updates
all five configurations to the selected density. If density changes, it retains
the baseline's original configuration, predictions, metrics, and diagnostics
in `seed_3000/initial_evaluation/` before regenerating derived evaluation files.
Training records, metadata, and model weights remain unchanged.
The separate `scripts/run_experiments/run_burgers_sigma_tuning.py` retains
`ACTION = "summarize"`; it reads artifacts without training or writing files.
After a successful check, select `ACTION = "run"` to execute the prepared
continuation: baseline seeds 3001–3002 and all three seeds for each positive
candidate. `BASELINE_SEEDS` selects launch seeds only; archived plans and summary
comparisons retain all three seeds. Keep the preparation revision fixed during
training; approved expansion configurations and execution selections can be
resolved under this protocol and committed with the completed results. Outputs use
`output/burgers/nu_0p001/sigma_*/<method>/seed_<seed>/`.
See [study context](../../../../docs/research_context.md),
[results](../../../../docs/results.md), and
[artifact meanings](../../../../docs/output_format.md).
