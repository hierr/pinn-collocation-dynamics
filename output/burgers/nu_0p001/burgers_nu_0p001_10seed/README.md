# Burgers ν = 0.001: frozen ten-seed evaluation

Prepared evaluation: **50 planned runs**, comprising five methods on paired
seeds 4000–4009. No training has started. Freeze **sigma = 1e-6** for diffusion,
selected on separate development seeds 3000–3002 in tuning checkpoint
`56ea078665bb85d73a5d02f26b83b366df7ca570` on `hyperparameter-tuning`.

## Design and selection

The [frozen configuration](../../../../config/burgers_nu_0p001_10seed.yaml)
compares fixed Hammersley, upstream random resampling, RAD, PACMANN, and PACMANN
with diffusion, in that order. Network architecture, optimizer settings,
point budgets, constraint treatment, and evaluation grids match the preceding
frozen comparisons. Viscosity, reference data, and selected sigma differ.
Only interior physics points adapt; the 80 boundary and 160 initial constraints
remain fixed. Each run starts with 2,500 interior points.

The development search completed 21 runs: sigma zero plus six positive
candidates, each on three paired seeds. Its initial boundary winner triggered
one approved lower expansion. The final positive winner, `sigma = 1e-6`, is
interior, so no further expansion is required. Select by arithmetic mean final
solution relative L2; exact ties favor the smaller positive value. Sigma zero
is a comparison baseline only, even when it has lower error. At the selected
value, development means were 44.16674% solution relative L2 and 2.949592
residual RMS, versus 41.75898% and 2.931608 for sigma zero. Errors were large
and highly variable; these results do not establish a diffusion benefit.
See the [accepted selection](../../../../docs/results.md#diffusion-selection-at-nu-0001).

## Reference and evaluation

Use the unchanged
[N4096 reference](../../../../data/burgers_dataset/burgers_nu_0p001_N4096.npz)
accepted on `dataset-validation` at `41529cb` and retained in the tuning
checkpoint above. Stored viscosity is exactly 0.001. Its SHA-256 is
`b628305506f4d3fcfc5052c725bfe595fee716bce21e951165a8ce0af85d89de`.
N4096 gives 4,097 Chebyshev spatial nodes. N2048–N4096 solution disagreement
was 0.00964855% relative L2 on a dense common grid; this supports the working
reference without bounding absolute error or certifying temporal convergence.
Reference data are used only for evaluation.

Final solution error uses 1,024 uniform spatial points and 199 native times
through `t = 0.99`. Interpolate reference values only in space and infer the
PINN directly at these coordinates. PDE-residual RMS uses a separate fixed
1,024-by-200 interior midpoint grid. The development baseline check, using
fixed weights, changed solution relative L2 by 0.005708% and residual RMS by
0.051008% when refining spatial sampling to 4,096 points. Both passed the
predeclared 1% relative threshold. This checks one model's spatial sampling;
it does not certify temporal resolution or every possible trained model.

Use final-model errors and retain every planned seed, including numerical
failures. Report per-seed outcomes, mean and sample SD, paired comparisons,
and supplementary trimmed means removing one minimum and one maximum
independently for solution L2 and residual RMS. Solution L2 is reported in
percent, paired solution changes in percentage points, and residual RMS in
its original units. Boundary and initial-condition RMSE are separate
diagnostics. Assess residual error at the solution-selected sigma and do not
retune on evaluation seeds. Model snapshots support evaluation, not exact
training resume.

## Reproduction and execution conditions

Use the [documented environment](../../../../README.md#setup-and-execution).
In `scripts/run_experiments/run_burgers_campaign.py`, select
`CONFIG_PATH = ROOT / "config/burgers_nu_0p001_10seed.yaml"` and the five methods
in the order above. From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
venv/bin/python -u -m scripts.run_experiments.run_burgers_campaign
```

The runner executes ten concurrent seeds per method, with one numerical CPU
thread per worker, on one NVIDIA GeForce RTX 5070 (12 GB) using float64.
Method batches run sequentially, including final evaluation, before the next
starts. Training time includes compilation and optimizer work and excludes
post-processing. It reflects ten-way GPU concurrency and is not directly
comparable with tuning timings. DeepXDE's batched L-BFGS driver can exceed
configured limits; report actual optimizer work from `training.json`.

All five methods and ten seeds remain in the frozen YAML. To execute one
method batch at a time, select only that method in the runner's `METHODS` list.
The runner preflights selected destinations and never overwrites an existing
run. External interruptions follow the repository recovery policy; no
favorable-error retries or replacement seeds are allowed.

Retain artifacts under
`output/burgers/nu_0p001/burgers_nu_0p001_10seed/<method>/seed_<seed>/`.
Per-seed and aggregate outcomes will be added here after execution.
Study-level comparisons belong in [results](../../../../docs/results.md);
see the [study context](../../../../docs/research_context.md) and
[artifact dictionary](../../../../docs/output_format.md).
