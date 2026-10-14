# Burgers sigma tuning: burgers_sigma_tuning_01_baseline

- [Study phase](../../docs/research_context.md#stage-1-original-benchmark-reproduction-and-diffusion-extension)
- [Results and setting directories](../../docs/results.md#original-benchmark-burgers-sigma-tuning)
- [Output dictionary](../../docs/output_format.md)

## Comparison design

Select a diffusion scale for later evaluation on the original PACMANN Burgers
benchmark. Compare PACMANN against diffusion candidates with common tuning seeds
and unchanged problem, network, training, and coordinate-Adam settings; only
diffusion varies. The selected YAMLs are listed in
[`CONFIG_PATHS`](../../scripts/run_experiments/run_burgers_sigma_tuning.py); each seed directory
archives its executed configuration and source revision.

After all planned runs and diagnostics complete, select the lowest mean final
relative L2 over the original reference grid; exact ties favor smaller sigma,
including zero. A positive winner at either search boundary requires review.
Report variability and paired consistency. This is exploratory tuning on the
same reference grid used for later seed evaluation, not a held-out-data test or
a sharpness comparison. The noise convention is defined in the
[output dictionary](../../docs/output_format.md#configuration-provenance-and-status).

## Execution

From the repository root, set `ACTION = "run"` in the tuning script and execute:

```bash
venv/bin/python -m scripts.run_experiments.run_burgers_sigma_tuning
```

The recorded sweep used the original module path `scripts.run_burgers_sigma_tuning`.
Set `ACTION = "summarize"` to regenerate its comparison from saved artifacts
without training. Existing run destinations are protected from overwriting.
