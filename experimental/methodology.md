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
