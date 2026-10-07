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
