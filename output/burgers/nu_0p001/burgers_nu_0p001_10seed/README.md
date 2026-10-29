# Burgers ν = 0.001: frozen ten-seed evaluation

Completed evaluation: **all 50 planned runs**, comprising five methods on paired
seeds 4000–4009, with finite outcomes and no reported execution failures.
Diffusion used frozen **sigma = 1e-6**,
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
in the order above. The retained run directories are occupied. For a new
repetition, copy the YAML with a new `campaign` identifier, retain all scientific
settings and seeds, and select the copy in `CONFIG_PATH`. From the repository root:

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
All runs record source revision `5a1844bfa9ddb51a49efcb0910bda6838347b523`, the
same dependency versions, CUDA execution, and reference fingerprint. Restoring
each final checkpoint reproduced saved predictions exactly and all five error
metrics within `1e-12 + 1e-10 * abs(saved)`. Initial point sets match across
methods for each seed, constraints remain fixed, and physics points remain
unchanged during L-BFGS. Fixed Hammersley points remain unchanged throughout.

Study-level comparisons belong in [results](../../../../docs/results.md);
see the [study context](../../../../docs/research_context.md) and
[artifact dictionary](../../../../docs/output_format.md).

## Per-seed outcomes

Solution L2 is in percent. Aggregates use all ten seeds and sample SD.
Supplementary trimmed means remove one minimum and one maximum independently
for solution L2 and residual RMS; no seed is removed from the primary summaries.

### Fixed Hammersley

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 0.157172 | 63.93833 | 3.277385 | 0.001225 | 0.038014 | 8800.7 |
| 4001 | 0.117012 | 55.16834 | 5.626318 | 0.001327 | 0.111101 | 8806.9 |
| 4002 | 0.172483 | 66.98029 | 3.989999 | 0.002108 | 0.038375 | 8802.0 |
| 4003 | 0.0618995 | 40.12524 | 3.911104 | 0.001464 | 0.090585 | 8801.5 |
| 4004 | 0.0817053 | 46.09984 | 2.459738 | 0.001694 | 0.105185 | 8804.6 |
| 4005 | 0.0605259 | 39.67755 | 2.387643 | 0.000128 | 0.001646 | 8793.2 |
| 4006 | 0.0571296 | 38.54826 | 3.013233 | 0.000292 | 0.005505 | 8811.0 |
| 4007 | 0.0821014 | 46.21144 | 3.044721 | 0.001700 | 0.045281 | 8797.7 |
| 4008 | 0.26647 | 83.25263 | 5.874944 | 0.001398 | 0.073245 | 8803.7 |
| 4009 | 0.0354019 | 30.34503 | 4.198030 | 0.001510 | 0.022221 | 8785.4 |
| Mean ± sample SD | 0.10919 ± 0.070989 | 51.03469 ± 16.17762 | 3.778311 ± 1.206309 | 0.001285 ± 0.000618 | 0.053116 ± 0.039805 | 8800.7 ± 7.2 |
| 10% trimmed mean | — | 49.59366 | 3.690066 | — | — | — |

### Random resampling

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 0.0154591 | 20.05237 | 1.972577 | 0.001102 | 0.020909 | 8761.2 |
| 4001 | 0.18807 | 69.94131 | 5.027258 | 0.002069 | 0.065240 | 8766.2 |
| 4002 | 0.0265321 | 26.27001 | 0.767609 | 0.000782 | 0.004062 | 8770.2 |
| 4003 | 0.099685 | 50.92009 | 1.702033 | 0.000608 | 0.014435 | 8750.0 |
| 4004 | 0.239189 | 78.87593 | 8.645658 | 0.001870 | 0.047538 | 8810.7 |
| 4005 | 0.0250115 | 25.50610 | 2.276996 | 0.001147 | 0.035685 | 8765.8 |
| 4006 | 0.0204048 | 23.03778 | 1.175823 | 0.000238 | 0.001238 | 8770.6 |
| 4007 | 0.0804622 | 45.74778 | 2.792916 | 0.000593 | 0.013544 | 8770.8 |
| 4008 | 0.153895 | 63.26827 | 2.998328 | 0.000817 | 0.032455 | 8739.3 |
| 4009 | 0.013178 | 18.51393 | 1.458518 | 0.000697 | 0.025812 | 8773.1 |
| Mean ± sample SD | 0.0861886 ± 0.0820231 | 42.21336 ± 22.60381 | 2.881772 ± 2.353171 | 0.000992 ± 0.000578 | 0.026092 ± 0.019856 | 8767.8 ± 18.5 |
| 10% trimmed mean | — | 40.59296 | 2.425556 | — | — | — |

### RAD

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 0.296027 | 87.74858 | 8.084387 | 0.002741 | 0.137513 | 9062.1 |
| 4001 | 0.210403 | 73.97764 | 4.023017 | 0.003794 | 0.118214 | 9059.7 |
| 4002 | 0.0725633 | 43.44428 | 2.575166 | 0.001622 | 0.123186 | 9049.7 |
| 4003 | 0.220404 | 75.71530 | 3.462165 | 0.003874 | 0.098758 | 9035.9 |
| 4004 | 0.125321 | 57.09354 | 4.226288 | 0.001147 | 0.022808 | 9029.5 |
| 4005 | 0.0846132 | 46.91299 | 2.808657 | 0.003340 | 0.056142 | 9060.2 |
| 4006 | 0.0143165 | 19.29712 | 0.208708 | 0.000391 | 0.012512 | 9063.3 |
| 4007 | 0.225524 | 76.58965 | 5.367015 | 0.005199 | 0.121864 | 9040.8 |
| 4008 | 0.0727554 | 43.50176 | 2.063159 | 0.000876 | 0.014202 | 9044.0 |
| 4009 | 0.0529087 | 37.09691 | 1.023046 | 0.001490 | 0.032313 | 9055.5 |
| Mean ± sample SD | 0.137484 ± 0.0935419 | 56.13778 ± 21.71956 | 3.384161 ± 2.249354 | 0.002447 ± 0.001574 | 0.073751 ± 0.050931 | 9050.1 ± 12.0 |
| 10% trimmed mean | — | 56.79151 | 3.193564 | — | — | — |

### PACMANN

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 0.0422429 | 33.14752 | 2.317639 | 0.000467 | 0.002533 | 9069.1 |
| 4001 | 0.259185 | 82.10677 | 7.674714 | 0.004212 | 0.111286 | 9079.8 |
| 4002 | 0.164303 | 65.37279 | 3.015101 | 0.006232 | 0.158433 | 9076.2 |
| 4003 | 0.0621226 | 40.19751 | 3.012430 | 0.002242 | 0.005713 | 9078.2 |
| 4004 | 0.0935032 | 49.31595 | 3.197740 | 0.002225 | 0.014756 | 9073.0 |
| 4005 | 0.13355 | 58.93816 | 3.113312 | 0.000473 | 0.042014 | 9042.6 |
| 4006 | 0.243861 | 79.64254 | 5.523190 | 0.002802 | 0.080864 | 9083.7 |
| 4007 | 0.00843629 | 14.81324 | 1.654348 | 0.001083 | 0.007114 | 9066.9 |
| 4008 | 0.0857531 | 47.22794 | 2.439986 | 0.002187 | 0.145138 | 9053.6 |
| 4009 | 0.0162918 | 20.58539 | 3.258315 | 0.004683 | 0.042877 | 9070.3 |
| Mean ± sample SD | 0.110925 ± 0.088512 | 49.13478 ± 22.87606 | 3.520678 ± 1.770670 | 0.002661 ± 0.001884 | 0.061073 ± 0.059350 | 9069.3 ± 12.6 |
| 10% trimmed mean | — | 49.30347 | 3.234714 | — | — | — |

### PACMANN + diffusion

| Seed | Solution MSE | Solution L2 (%) | Residual RMS | Boundary RMSE | Initial RMSE | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 0.143431 | 61.07956 | 4.180329 | 0.001475 | 0.046500 | 9059.3 |
| 4001 | 0.10216 | 51.54844 | 3.196463 | 0.001006 | 0.036143 | 9061.6 |
| 4002 | 0.19363 | 70.96765 | 5.359139 | 0.002070 | 0.051456 | 9074.3 |
| 4003 | 0.123579 | 56.69517 | 4.089219 | 0.001931 | 0.026186 | 9072.8 |
| 4004 | 0.227034 | 76.84577 | 4.193044 | 0.002637 | 0.164534 | 9071.8 |
| 4005 | 0.0835993 | 46.63108 | 3.288973 | 0.001208 | 0.025790 | 9039.9 |
| 4006 | 0.231702 | 77.63170 | 7.401824 | 0.003602 | 0.096347 | 9059.0 |
| 4007 | 0.332704 | 93.02576 | 11.083931 | 0.009900 | 0.109890 | 9063.6 |
| 4008 | 0.00422482 | 10.48282 | 1.578566 | 0.000429 | 0.001610 | 9048.6 |
| 4009 | 0.147746 | 61.99151 | 4.242948 | 0.002600 | 0.038625 | 9057.7 |
| Mean ± sample SD | 0.158981 ± 0.0917616 | 60.68995 ± 22.40796 | 4.861444 ± 2.652718 | 0.002686 ± 0.002696 | 0.059708 ± 0.049149 | 9060.9 ± 10.8 |
| 10% trimmed mean | — | 62.92386 | 4.493992 | — | — | — |

## Optimizer work and interpretation

Fixed Hammersley has no sampling updates; the other methods have 700 per seed.
Every run completed 35,000 Adam steps. L-BFGS work below is mean ± sample SD;
random-resampling seed 4004 completed 15,986 iterations, while all other runs
completed 15,000. Shared configured limits do not imply identical realized work.

| Method | L-BFGS iterations | L-BFGS function evaluations |
| --- | ---: | ---: |
| Fixed Hammersley | 15000.0 ± 0.0 | 16668.6 ± 216.7 |
| Random resampling | 15098.6 ± 311.8 | 16739.8 ± 697.8 |
| RAD | 15000.0 ± 0.0 | 16349.2 ± 118.1 |
| PACMANN | 15000.0 ± 0.0 | 16795.5 ± 213.4 |
| PACMANN + diffusion | 15000.0 ± 0.0 | 16790.4 ± 251.2 |

Random resampling has the lowest mean and trimmed-mean solution and residual
errors. All 50 solution relative L2 errors are large, ranging from 10.48282%
to 93.02576%. At the selected sigma, diffusion raises PACMANN's mean solution
error by 11.555165 percentage points and residual RMS by 1.340766; it improves
four of ten paired solution errors and two of ten residual errors. The
[study-level comparison](../../../../docs/results.md#frozen-evaluation-at-nu-0001)
retains these unfavorable outcomes, variability, and the seven paired contrasts.
Reference limitations and unmeasured feature-region errors limit broader
sharpness conclusions.
