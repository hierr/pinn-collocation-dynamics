# Burgers: frozen-setting ten-seed comparison

**Frozen protocol; no evaluation runs have started.** The
[configuration](../../config/burgers_nu_0p0031830989_10seed.yaml) contains the full
five-method study. Accepted numerical implementation: `6934073`.

## Design and comparisons

Evaluate fixed Hammersley, upstream random resampling, RAD, PACMANN, and PACMANN
with diffusion on ten new paired seeds: fifty planned runs at `nu = 0.01/pi`
using the validated generated N1024 reference. This evaluates methods at one
viscosity; it does not test a sharpness effect or reproduce the original paper's
unequal-population setup.

The primary contrast is diffusion versus PACMANN for final solution relative L2,
using `sigma=1e-4` selected on separate development seeds. Additional contrasts
compare RAD, PACMANN, and PACMANN with diffusion against fixed Hammersley and
random resampling. Assess the same contrasts for independent residual RMS without
selecting another sigma.
Boundary and initial-condition RMSE are constraint diagnostics.

Use the YAML's planned seed list. Match initial weights,
Hammersley points, fixed constraints, PDE point budgets, and optimizer settings
across methods within each seed. Sampling settings are fixed before evaluation:
upstream sampling settings for the baselines and PACMANN, and development-selected
sigma for diffusion. Disclose this tuning asymmetry. Retain the development sweep's
L-BFGS behavior and compare actual iterations, function evaluations, and runtime.

Report every seed's final-model outcomes, means and sample SD, paired absolute
error changes, and counts improved for each declared contrast. Negative paired
changes mean lower error; exact ties count separately. Express solution L2 as
percent and its paired changes as percentage points; residual changes remain in
residual units. Do not retune, select history minima, or replace unfavorable
seeds. Report a worker failure after all started workers finish; retain every
run's artifacts and review failures before launching another method. Follow the
[recovery policy](../../AGENTS.md#execution-recovery). Do not present a complete
ten-seed mean from a successful subset; report missing/failed outcomes and the
available paired count explicitly.

## Execution

Pull the committed `main` on `hier-wsl`, use its existing `.venv`, and verify CUDA
execution and GPU availability. In `scripts/run_experiments/run_burgers_campaign.py`,
`CONFIG_PATH` selects the configuration above and `METHOD` selects one method.
Start with `pacmann_adam`, then `pacmann_diffusion`; the remaining methods are
`fixed_hammersley`, `random_resampling`, and `rad`. Change only the method
selector between launches; keep the numerical YAML fixed.

Each invocation launches all ten seeds concurrently in independent processes,
then exits when all workers finish. Only one method batch uses the GPU at a time.
The launcher checks the selected method's destinations before starting; completed
other methods do not block a later launch. It does not skip or retry runs.
Workers use one CPU thread and set OMP/MKL/OpenBLAS thread limits to one, as
described in the [environment instructions](../../README.md#setup-and-execution).

Report mean and sample SD of all ten `training_seconds` values for each method,
describing the timing condition as up to ten concurrent seeds on one GPU. These
recorded times include competition with other workers, including workers doing
final evaluation. Do not combine them with timings from serial tuning runs.

From the remote checkout, launch detached so SSH disconnection does not stop it:

```bash
cd /home/hiero/pinn-collocation-dynamics
tmux new-session -d -s burgers_pacmann_adam \
  '.venv/bin/python -u -m scripts.run_experiments.run_burgers_campaign > /tmp/burgers_pacmann_adam.log 2>&1'
```

Use a method-specific session/log name for later launches.
Per-run artifacts are retained here; the driver log in `/tmp` is disposable.
Completed training still needs predictions, metrics, and diagnostics before
comparison. No training resume or automatic retry is implied.

[Study controls](../../docs/research_context.md#accepted-controls-and-evaluation-policy)
· [Development selection](../../docs/results.md#n1024-diffusion-development-selection)
· [Artifact dictionary](../../docs/output_format.md)
· [Environment](../../README.md#setup-and-execution)
· [Optimizer semantics](../../references/upstream.md#controlled-study-choices)
