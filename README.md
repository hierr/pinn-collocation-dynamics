# PINN Collocation Dynamics

Residual-guided collocation and stochastic diffusion for PINNs: evaluating
performance across smooth and sharp-feature PDE regimes.

This research investigates adaptive collocation methods for physics-informed
neural networks, with a possible publication as the longer-term objective.

## Research questions

- When does residual-gradient adaptive collocation improve on low-discrepancy
  sampling, and how does that benefit change across smooth and sharp regimes?
- Does adding stochastic diffusion to PACMANN help or hurt under those conditions?

Sharp solution gradients and gradients of the squared PDE residual are distinct.
The questions above are hypotheses to investigate, not established claims.

## First milestone

Reproduce the upstream Burgers benchmark with PACMANN and its Hammersley
baseline, then add diffusion as a controlled change. DeepXDE with the PyTorch
backend is the intended implementation. The pinned upstream Burgers scripts,
requirements, license, and file manifest are imported under
`references/pacmann/burgers/`. The ignored reference dataset can be restored using
[source provenance instructions](references/upstream.md#restore-the-dataset).
No maintained trainer or environment exists yet; select and lock dependencies
before implementation. The import is not a completed scientific reproduction.

## Structure

```text
src/                    Minimal training code and sampler callbacks, when added
config/                 Experiment YAML files, when added
experimental/           Research context, methodology, and experiment records
references/             Source provenance and upstream implementation pin
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
[working agreements](AGENTS.md).
