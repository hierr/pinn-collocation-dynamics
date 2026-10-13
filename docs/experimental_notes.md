# Experimental notes

Record scientific campaign protocols here before execution, then add outcomes,
interpretation, limitations, material deviations, decisions, and next steps.
Keep entries at campaign level; generated artifacts hold individual seed records.
Tuning is scientific development and must be distinguished from frozen-setting
comparisons. See [research context](research_context.md) for the study plan,
[methodology](methodology.md) for reusable procedures, and [results](results.md)
for concise comparisons.

<a id="burgers-pilot-01"></a>

Engineering readiness checks are excluded from this scientific record. Their
existing artifacts are retained; this anchor preserves earlier links.

<a id="burgers-sigma-tuning-01"></a>

## 2026-10-08: Original-benchmark Burgers sigma tuning

Scientific development for Stage 1. All 15 planned runs and diagnostics completed.
The predeclared lowest-mean criterion selects sigma `1e-4`: mean final relative
L2 was 0.04710% (sample SD 0.00793%), compared with 0.07785% (sample SD 0.02905%)
for ordinary PACMANN. The mean paired change was -0.03075 percentage points, with
improvement in two of three seed pairs. The `1e-3` results include one much larger
error (28.1237%), retained as a completed run; this setting's mean and variability
are correspondingly poor. See the [results table](results.md#original-benchmark-burgers-sigma-tuning)
and [campaign artifacts](../output/burgers_sigma_tuning_01_baseline/README.md).
This selects a candidate for separate evaluation on the original benchmark; it
does not test sharpness dependence or establish a general diffusion benefit.
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004385
### Objective and numerical change

Objective: estimate whether Gaussian exploration improves PACMANN's final solution
accuracy on the original Burgers benchmark, and select a candidate noise scale
for later evaluation. This is tuning of a modified sampler, not reproduction of
an upstream diffusion method or a test of benefits across sharpness regimes.
The working hypothesis is that a nonzero sigma may improve mean final relative
L2 over ordinary PACMANN. Sigma zero remains a valid selection.

During each of K coordinate-Adam steps, first perform the original ascent update,
then add `sigma * L / sqrt(K) * xi`, with independent standard-normal entries in
`xi` and `L = [x_max - x_min, t_max - t_min]`. Sigma is dimensionless and describes
the accumulated noise component's standard deviation as a fraction of each
domain width per movement event, before boundary handling. This statement is
about the noise component, not the total trajectory: gradients depend on the
noisy coordinates and replacement changes escaped trajectories. It is not a
physical Burgers viscosity or a claim of a particular continuous-time SDE.

Use an independent persistent NumPy generator initialized with
`SeedSequence([seed, 1])`. Draw noise only when sigma is positive. Preserve fresh
coordinate-Adam moments at each event, movement every 50 network Adam steps,
15 coordinate steps per event, and upstream uniform replacement of escaped
points only after those 15 steps. Keep constraints fixed; do not move points
during L-BFGS. All configurations share initial points and network weights for
each seed. Positive noise may change subsequent escaped-point replacement draws.

### Configurations and controls

The planned sweep uses seeds 2000, 2001, and 2002 for each setting:

| Sigma | Method | Configuration | Campaign |
| ---: | --- | --- | --- |
| 0 | PACMANN-Adam | `config/burgers_sigma_baseline.yaml` | `burgers_sigma_tuning_01_baseline` |
| 1e-5 | PACMANN + diffusion | `config/burgers_sigma_1e-5.yaml` | `burgers_sigma_tuning_01_1e-5` |
| 1e-4 | PACMANN + diffusion | `config/burgers_sigma_1e-4.yaml` | `burgers_sigma_tuning_01_1e-4` |
| 1e-3 | PACMANN + diffusion | `config/burgers_sigma_1e-3.yaml` | `burgers_sigma_tuning_01_1e-3` |
| 1e-2 | PACMANN + diffusion | `config/burgers_sigma_1e-2.yaml` | `burgers_sigma_tuning_01_1e-2` |

Controls: unchanged checksummed reference data and viscosity `0.01/pi`, bounds
`[-1, 1] x [0, 0.99]`, 2,500 initial interior points, 80 boundary and 160 initial
constraints, four 64-neuron tanh layers, float64, the pinned environment, and
common host GPU. Both adaptive variants retain 2,740 initial PDE points and
replace them with 2,500 interior anchors at the first movement event. Coordinate
Adam uses step size 1e-5, betas 0.9/0.999, and epsilon 1e-7. Only diffusion and
its sigma vary; no Hammersley runs or other samplers are part of this sweep.

Each run uses five phases of 7,000 network Adam steps at learning rate 1e-3,
followed by L-BFGS with a configured limit of 3,000 iterations per phase and the
unchanged upstream-reproduction options in the listed YAMLs. Record history every
100 steps and retain actual optimizer counts, including L-BFGS overshoot.
Expect 700 movement events per successful
run. Budgets are matched by configured settings, not guaranteed equal actual
optimizer work. The sweep contains 15 full-budget training runs; diffusion and
convergence can change actual runtime.

### Execution and recovery

Execute baseline first, then increasing sigma, with seeds in their listed order,
sequentially on the host GPU. The no-argument tuning script selects configurations
and `ACTION` at its top. Before scientific launch, set `ACTION = "run"` and commit
the source/configuration checkpoint. Launch using
`venv/bin/python -m scripts.run_burgers_sigma_tuning`. It checks the entire sweep
before starting, retains all scientific run artifacts, and creates short campaign
README links without overwriting existing ones. Stop on the first training or
analysis failure. Retain planned seeds and numerical failures; do not silently
retry numerical failures, replace seeds, or change budgets. External interruptions
follow the [operational recovery policy](methodology.md#execution-interruptions-and-recovery)
and do not constitute scientific failure evidence. An unchanged same-seed
restart or regeneration of missing analysis is recovery within the approved
scope; preserve completed outputs and resolve occupied destinations before
relaunching. The runner does not perform this recovery automatically.

### Evaluation and selection

Primary selection metric: mean final relative L2 on the complete original reference
grid across the three tuning seeds. The final model is evaluated at the configured
training endpoint, not selected by test error. The reference grid is used for
tuning; later seed evaluation on this same benchmark is not a held-out dataset
test. Secondary observations are final MSE, upstream L2 at minimum recorded
training loss, histories, actual optimizer work, movement events, training time,
and existing error diagnostics. Adaptive training losses do not rank sigmas.

Post-processing with `ACTION = "summarize"` reads archived settings, statuses,
metrics, and provenance without retraining or writing results. Report individual
seed errors, planned/completed/failed/pending execution counts, mean final L2, sample
standard deviation (denominator n-1), mean paired error changes versus PACMANN,
the count of improved seed pairs, and completed training time. Means and standard
deviations require all three seeds in the configuration to complete. Paired mean
changes require all three baseline and candidate seed pairs. Completed training
with missing diagnostics remains pending for this sweep until analysis is
regenerated. Execution counts do not by themselves classify scientific failures;
interpret the cause using the recovery policy. Reject comparisons with changed
archived settings or mismatched source, environment, data, or device. Do not rank
settings using only surviving
seeds after a failure.

Decision rule: only after all 15 runs and diagnostics complete successfully,
select the setting with the lowest mean final relative L2; exact ties favor
smaller sigma. A positive winner at 1e-5 or 1e-2 requires review before freezing
the setting; do not expand the search automatically. Report paired consistency
and variability even when the mean favors diffusion. Three tuning seeds provide
exploratory selection, not a superiority claim. Close or inconsistent outcomes
remain uncertain and must be described that way.

### Reporting and next decision

The completed tuning comparison and its interpretation are recorded in
[results](results.md). Freeze sigma `1e-4` as the candidate for the later
ten-seed evaluation on this benchmark. This choice follows the predeclared mean
criterion, despite improvement in only two of three paired seeds, and remains
specific to this Burgers case. Do not generalize it to other viscosities or PDE
families.

Freeze the selected sigma before the later ten-seed evaluation using seeds
1000–1009, which are excluded from sigma tuning. Seed 1000 has previously been
inspected during readiness checks; disclose that prior inspection when reporting
final evaluation. Prepare the final comparison protocol after the tuning decision.
The [parameterized stage](research_context.md#stage-2-parameterized-sharpness-study)
requires new reference solutions and its own development search; this single
viscosity search does not evaluate sharpness dependence.
