# PINN Collocation Dynamics

Residual-guided collocation and stochastic diffusion for PINNs: evaluating
performance across smooth and sharp-feature PDE regimes.

This research continues the PCS5022 project, is currently developed for the
PME5242 Physics-Informed Machine Learning coursework, and targets a possible
publication. The previous repository remains the historical experiment record;
this repository starts with a small, independent implementation.

## Research questions

- When does residual-gradient adaptive collocation improve on low-discrepancy
  sampling, and how does that benefit change across smooth and sharp regimes?
- Does adding stochastic diffusion to PACMANN help or hurt under those conditions?

Sharp solution gradients and gradients of the squared PDE residual are distinct.
The questions above are hypotheses to investigate, not established claims.

## First milestone

Reproduce the upstream Burgers benchmark with PACMANN and its Hammersley
baseline, then add diffusion as a controlled change. DeepXDE with the PyTorch
backend is the intended implementation. No trainer or environment is included
in this bootstrap; select and lock dependencies before implementation.

## Structure

```text
src/                    Minimal training code and sampler callbacks, when added
config/                 Experiment YAML files, when added
experimental/           Research context, methodology, and experiment records
references/             Source provenance and upstream implementation pin
coursework/PME5242/     Class-specific requirements and submission material
```

Use `data/` for reference data and
`output/<campaign>/<problem>/<method>/<seed>/` for generated run artifacts.
Both are ignored by Git. Before a campaign, document artifact storage and
retrieval; ignored local files alone are not a reproducibility archive.
Each run must archive its resolved configuration and must not overwrite an
existing run directory. Scientific settings belong in configurations.

Add functionality only when the next experiment requires it. Begin with phase
endpoints, final models, and selected collocation snapshots; exact resume and
campaign automation are deferred.

See [source provenance](references/upstream.md) and
[working agreements](AGENTS.md). A future submission tag can preserve the
PME5242 checkpoint while publication work continues in this repository.
