# Experimental notes

The Burgers full-budget single-seed pilot is complete.

For each material experiment, record its campaign-specific protocol before
execution, followed by results, observations, limitations, decisions, and next
steps. General design and evaluation guidelines are in `methodology.md`.

<a id="burgers-pilot-01"></a>

## 2026-10-06: Burgers full-budget pilot — burgers_pilot_01

[Campaign artifacts and output dictionary](../output/burgers_pilot_01/README.md).

Objective: assess numerical reproduction readiness before a ten-seed campaign.
This paired single-seed pilot does not test diffusion or the sharpness hypothesis.
Controls: original checksummed reference grid, pinned environment, float64,
four 64-neuron tanh layers, common Hammersley initialization, fixed constraints,
and unchanged PACMANN settings. Changed variable: fixed Hammersley versus
PACMANN-Adam sampling, retaining the released-code population difference.

The approved protocol uses seed 1000 and `config/burgers_pilot.yaml`. Run fixed
Hammersley first, then PACMANN-Adam, sequentially on the host GPU. Each method
uses five phases of 7,000 network Adam steps at 1e-3 and L-BFGS configured for
3,000 iterations, with history recorded every 100 steps.

Preserve released-code behavior: 2,740 initial PDE coordinates, PACMANN's
reduction to 2,500 interior anchors, 240 fixed constraints, and possible L-BFGS
iteration overshoot. Record actual optimizer endpoints. This is code reproduction,
not a matched-population or strict-budget comparison.

Evaluation uses the complete original reference grid and saved configurations;
no model or setting is selected using test errors. The primary diagnostic is
relative L2 at minimum recorded training loss, supplemented by final MSE/L2,
minimum observed errors, actual optimizer counts, convergence history, and timing.
Published Table 2 means (Hammersley 0.61%, PACMANN-Adam 0.07%) provide ten-run
context, not single-seed acceptance thresholds or evidence of superiority.

Require finite losses, predictions, and metrics; identical initial designs;
fixed constraints; no movement during L-BFGS; unchanged Hammersley points;
700 PACMANN updates; complete phase artifacts; and final-weight reload agreement
at rtol 1e-10 / atol 1e-12. Preserve failed runs and planned seeds; do not silently
replace or retry them. Apply the general run-retention policy in methodology.
Review accuracy, convergence, and limitations before separately authorizing the
remaining nine seeds. Diffusion is deferred.

Results: both methods completed on the host GPU with the protocol/source at
commit `eba6c86`. Tracked source and configuration were clean at run start;
a user-added PACMANN paper PDF was untracked.
Hardware: NVIDIA GeForce RTX 5050 Laptop GPU, driver 595.91.07. The original
reference dataset and imported source hashes remain unchanged.

Hammersley achieved primary L2 error 0.182490% at step 30,281; its final L2 was
0.202476% and final MSE 1.547251e-6. PACMANN achieved primary and final L2
0.0737324% at step 50,620, with final MSE 2.051779e-7. Minimum observed test L2
was 0.0862134% for Hammersley and 0.0440476% for PACMANN, reported separately
because test-error selection is not our primary criterion. Both final weight
reloads reproduced predictions exactly.

All acceptance invariants passed: identical initial designs; 240 constraints
fixed throughout; unchanged Hammersley PDE population; no L-BFGS movement;
700 PACMANN updates (140 per Adam phase); complete phase weights/points; and
finite archived histories, predictions, and metrics. Both methods performed
35,000 Adam steps. Actual L-BFGS counts by phase were 3,000 / 3,000 / 3,281 /
3,611 / 3,271 for Hammersley (16,163 total), and 3,000 / 3,000 / 3,000 /
3,250 / 3,370 for PACMANN (15,620 total). This is not a strict equal-budget
comparison, and the released PDE population difference is also retained.

Training-plus-compilation timings were 978.1 seconds for Hammersley and 906.1
seconds for PACMANN. These timings exclude evaluation reload, post-processing,
and archival; a single pair does not establish a runtime advantage or reproduce
the paper's hardware-dependent timing statistics.

Convergence was non-monotonic after optimizer resets. PACMANN showed large
within-Adam reference-error fluctuations, followed by recovery in L-BFGS;
for example, its second Adam endpoint had about 28.5% L2 error before the next
L-BFGS endpoint recovered to about 0.0924%. Low training loss and lowest test
error did not select the same step in Hammersley. These observations motivate
reporting full histories rather than only one favorable metric.

Final-grid maximum absolute error was 0.0248566 for Hammersley near
(x, t) = (-0.00392157, 0.83), and 0.00947801 for PACMANN near
(0.00392157, 0.50), both close to the steep central transition. Per-time errors,
convergence, solution slices, and final interior point plots are stored as
`diagnostics.json` / `diagnostics.png` alongside each run. Post-processing uses
archived coordinates and predictions; no new random evaluation set is used.

The primary PACMANN error is close to the published 0.07% ten-run mean, while
this Hammersley seed is below the published 0.61% mean (Table 2,
https://doi.org/10.1016/j.cma.2025.118723). This provides a credible pilot outcome,
not reproduction of the means/standard deviations or evidence of superiority.
No seed was replaced and neither run failed. Diffusion and sharpness benefits
remain untested.

Decision: the implementation passes the pilot gates. Stop for review before
expanding to ten seeds. Preserve the released-code protocol for reproduction;
address population matching and strict optimizer budgets before later controlled
method comparisons. Scientific run directories are retained under `output/burgers_pilot_01`
and will be tracked directly in Git. No off-machine copy is claimed.

<a id="burgers-sigma-tuning-01"></a>

## 2026-10-08: Burgers sigma tuning — prepared protocol

Status: protocol, configurations, sampler, and execution support are prepared.
The scientific sweep has not started, and no sigma has been selected. Launch is
a separate checkpoint after review and a committed source/configuration revision.

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
unchanged pilot options. Record history every 100 steps and retain actual optimizer
counts, including L-BFGS overshoot. Expect 700 movement events per successful
run. Budgets are matched by configured settings, not guaranteed equal actual
optimizer work. Fifteen runs suggest approximately four hours based on the
previous PACMANN pilot, but diffusion and convergence can change this estimate.

Execute baseline first, then increasing sigma, with seeds in their listed order,
sequentially on the host GPU. The no-argument tuning script selects configurations
and `ACTION` at its top. Before scientific launch, set `ACTION = "run"` and commit
the source/configuration checkpoint. Launch using
`venv/bin/python -m scripts.run_burgers_sigma_tuning`. It checks the entire sweep
before starting, retains all scientific run artifacts, and creates short campaign
README links without overwriting existing ones. Stop on the first training or
analysis failure; retain planned seeds and failed runs without silent retries,
replacement seeds, or changes to budgets.

Primary selection metric: mean final relative L2 on the complete original reference
grid across the three tuning seeds. The final model is evaluated at the configured
training endpoint, not selected by test error. The reference grid is used for
tuning; later seed evaluation on this same benchmark is not a held-out dataset
test. Secondary observations are final MSE, upstream L2 at minimum recorded
training loss, histories, actual optimizer work, movement events, training time,
and existing error diagnostics. Adaptive training losses do not rank sigmas.

Post-processing with `ACTION = "summarize"` reads archived settings, statuses,
metrics, and provenance without retraining or writing results. Report individual
seed errors, planned/completed/failed/pending counts, mean final L2, sample
standard deviation (denominator n-1), mean paired error changes versus PACMANN,
the count of improved seed pairs, and completed training time. Means and standard
deviations require all three seeds in the configuration to complete. Paired mean
changes require all three baseline and candidate seed pairs. Completed training
with missing diagnostics remains pending for this sweep until analysis is
regenerated. Reject comparisons with changed archived settings or mismatched
source, environment, data, or device. Do not rank settings using only surviving
seeds after a failure.

Decision rule: only after all 15 runs and diagnostics complete successfully,
select the setting with the lowest mean final relative L2; exact ties favor
smaller sigma. A positive winner at 1e-5 or 1e-2 requires review before freezing
the setting; do not expand the search automatically. Report paired consistency
and variability even when the mean favors diffusion. Three tuning seeds provide
exploratory selection, not a superiority claim. Close or inconsistent outcomes
remain uncertain and must be described that way.

After scientific execution, put the printed comparison table in `results.md`,
then add results, limitations, and the freeze or search-revision decision here.
If failures prevent completion, record their causes and review any proposed
protocol change before additional runs. Freeze the selected sigma before the
later ten-seed evaluation using seeds 1000–1009, which are excluded from sigma
tuning; seed 1000 has already been examined in the reproduction pilot. Other
viscosities or problems remain necessary to study transfer and sharpness effects.
