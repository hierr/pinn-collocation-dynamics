# Experimental notes

No scientific experiment runs have been completed.

For each material experiment, add its objective/hypothesis, controlled and changed
variables, metrics, reproducibility requirements, and decision criteria before
execution. Afterwards record results, limitations, conclusion, and next steps.

## 2026-10-06: Burgers full-budget pilot — burgers_pilot_01

Objective: assess numerical reproduction readiness before a ten-seed campaign.
This paired single-seed pilot does not test diffusion or the sharpness hypothesis.
Controls: seed 1000, original checksummed reference grid, pinned environment,
float64, four 64-neuron tanh layers, common Hammersley initialization, fixed
constraints, full five-phase training schedule, and unchanged PACMANN settings.
Changed variable: fixed Hammersley versus PACMANN-Adam sampling, retaining the
released-code population difference. Evaluation uses the complete original
reference grid and archived configurations; no model or setting is selected
using test errors. The primary diagnostic is L2 at minimum recorded training
loss, supplemented by final MSE/L2, minimum observed errors, actual optimizer
counts, convergence history, and timing.

Acceptance and archive policy follow methodology. Require 700 PACMANN events,
finite results, fixed constraints and L-BFGS points, identical initial designs,
complete phase artifacts, and reload agreement. Preserve failures. Compare
accuracy descriptively with published ten-run means, without a single-seed
pass/fail threshold or superiority claim. Stop for review after this pair.

Results: pending execution.
