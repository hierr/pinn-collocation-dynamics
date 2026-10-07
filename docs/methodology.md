# Research methodology

This document defines reusable guidelines for designing and evaluating the study's
experiments. Campaign-specific protocols and run records belong in
`experimental_notes.md`; concise outcome comparisons belong in `results.md`.
Document each campaign's objective, controls, changed variables, evaluation
protocol, and decision criteria in experimental notes before execution.

## Experiment design

State the research question and changed variables before running an experiment.
Distinguish reproduction of released code from a modified benchmark study, and
record upstream source pins and deviations. Investigate adaptive sampling and
diffusion benefits as open questions rather than assumed outcomes. Validate the
implementation and reference solution before a scientific campaign; reference
solutions must match the PDE parameters being studied.

## Controlled comparisons

Pair methods using the same planned seeds, network initialization, and initial
point design where applicable. Keep problem definition, network, precision,
constraints, and training settings fixed unless they are variables under study.
In matched sampling comparisons, move only physics points and keep boundary,
initial, and supervised constraints fixed. Match point and training budgets;
report actual optimizer work and disclose intentional reproduction deviations.

## Evaluation and interpretation

Evaluate solution error on fixed reference or probe sets independent of training
collocation points. Declare metrics and checkpoint-selection rules before runs;
distinguish final errors from training-loss-selected errors and minimum observed
test errors. Define feature regions independently of adaptive point placement
when studying sharp features, and report both global and feature-region errors.

Report variability across planned seeds, failures, actual computational effort,
and timing definitions. Retain failed runs rather than silently replacing seeds.
Aggregate only comparable completed evaluations and disclose missing runs.
Treat single-seed pilots as readiness checks, not reproduction of population
statistics or evidence of method superiority. Record limitations and decision
criteria before expanding a campaign or generalizing its conclusions.

## Reproducibility

Save each run's resolved configuration, seed, code revision, environment versions,
model weights, selected point snapshots, metrics, predictions, history, log, and
timing in a fresh directory. Never overwrite a prior run or alter original
reference data. Use saved configurations and fixed evaluation seeds for
post-processing. Record results, interpretation, and next steps in experiment
notes, with compact comparable tables in results.

## Reproduction environment

Use Python 3.12 with uv and the local `venv`. The authoritative dependency pins
are `pyproject.toml` and `uv.lock`: DeepXDE 1.14.0, PyTorch 2.7.1 with CUDA 12.8
wheels, NumPy 2.3.1, SciPy 1.16.0, Matplotlib 3.10.3, and PyYAML 6.0.2.
Retain this released-code stack for the initial upstream reproduction. Conda is
unnecessary for the current wheel-based environment. Use the PyTorch backend
and float64. Available local GPU hardware is an NVIDIA GeForce RTX 5050 Laptop
GPU; archive the actual device and installed versions with each scientific run.

Environment validation is a setup activity. Record final environment decisions
here; reserve experiment notes and results for scientific experiment runs.

## Run retention and retrieval

Track scientific run directories directly in Git, including failed scientific
runs. Retain configurations, metadata, logs, model weights, point snapshots,
predictions, histories, and diagnostics under the existing output hierarchy.
Retrieve these files from the repository at the corresponding commit; never
rerun into an existing directory. Reference datasets remain ignored and are
restored using the pinned provenance instructions. Compressed archives and
archive checksum sidecars are not required. Setup-validation and compatibility
outputs are disposable and excluded from scientific records and Git.
