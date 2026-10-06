# Experimental procedure

1. Agree on the question, comparison, controls, and evaluation criteria before
   implementation or material runs.
2. Record upstream source pins and all reproduction deviations.
3. Validate the implementation and reference solution before a campaign.
4. Archive each run's configuration, seed, code revision, environment versions,
   model, selected point snapshots, metrics, log, and timing in a fresh directory.
5. Evaluate on fixed independent reference/probe sets. Retain planned seeds and
   record failures rather than silently replacing them.
6. Record results, limitations, conclusions, and next steps; aggregate only
   comparable completed evaluations and disclose missing or failed runs.

The concrete Burgers reproduction protocol and artifact archive policy must be
approved before execution. Exact resume and large campaign automation are deferred.

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

## Burgers reproduction pilot

The approved first scientific run is one paired seed (1000), using
`config/burgers_pilot.yaml` and the existing runner. Run fixed Hammersley first,
then PACMANN-Adam, sequentially on the host GPU. Keep all full reproduction
settings: five phases of 7,000 network Adam steps at 1e-3 and L-BFGS configured
for 3,000 iterations, with history recorded every 100 steps.

Preserve released-code behavior, including 2,740 initial PDE coordinates,
PACMANN's reduction to 2,500 interior anchors, 240 fixed constraints, and
possible L-BFGS iteration overshoot. Record actual optimizer endpoints; this is
code reproduction, not a matched-population or strict-budget comparison.
Primary diagnostic: relative L2 at minimum recorded training loss. Also report
final MSE/L2 and minimum observed test errors as distinct statistics. The
published Burgers Table 2 means (Hammersley 0.61%, PACMANN-Adam 0.07%) are
context from ten runs, not single-seed acceptance thresholds.

Require finite losses/predictions/metrics, identical initial points across
methods, fixed constraints, no movement in L-BFGS, unchanged Hammersley points,
700 PACMANN updates, complete artifacts, and final-weight reload agreement at
rtol 1e-10 / atol 1e-12. Keep failed runs and planned seeds; do not silently
replace or retry them. Review accuracy, convergence, and limitations before
separately authorizing the remaining nine seeds. Diffusion is deferred.

### Pilot archive and retrieval

Retain the original ignored run directories. Create a fresh compressed archive
`output/archives/burgers_pilot_01.tar.gz` containing both runs (including failures
if any), the original dataset, source/provenance, pilot configuration, dependency
manifest/lock, and committed methodology. Include maintained source and tests
for local retrieval without depending on a remote checkout. Store a SHA-256
sidecar; verify the archive and byte-for-byte extraction into a temporary
directory. Retrieve by verifying the checksum before extracting into a fresh
directory, never over existing runs. This local policy provides no off-machine
backup; external storage must be agreed before a larger campaign.
