# Burgers reproduction pilot: burgers_pilot_01

- [Protocol, results, limitations, and next steps](../../docs/experimental_notes.md#burgers-pilot-01)
- [Comparison table](../../docs/results.md#burgers-reproduction-pilot--seed-1000)
- [Output file and field dictionary](../../docs/output_format.md)

Saved runs:

- [Fixed Hammersley, seed 1000](burgers/fixed_hammersley/seed_1000/)
- [PACMANN-Adam, seed 1000](burgers/pacmann_adam/seed_1000/)

Each seed directory's `config.yaml` records its executed configuration.

## Commands

Run from the repository root using the
[pinned reproduction environment](../../docs/methodology.md#reproduction-environment).
The pilot used seed 1000 (`--seed-index 0`), with fixed Hammersley followed by
PACMANN-Adam, sequentially on the host GPU:

```bash
venv/bin/python -m src.train --config config/burgers_pilot.yaml --method fixed_hammersley --seed-index 0
venv/bin/python -m src.train --config config/burgers_pilot.yaml --method pacmann_adam --seed-index 0
```

The training source revision was `eba6c86`; each run's `metadata.json` records
the full revision, dependency versions, reference dataset fingerprint, and device.
The saved run configurations match `config/burgers_pilot.yaml`, with the selected
method and seed added. Existing run directories are protected from overwriting;
use a configuration with a new campaign name for a new approved run.

Regenerate diagnostics for the saved runs with the current analysis script:

```bash
venv/bin/python scripts/analyze_burgers.py --run-dir output/burgers_pilot_01/burgers/fixed_hammersley/seed_1000
venv/bin/python scripts/analyze_burgers.py --run-dir output/burgers_pilot_01/burgers/pacmann_adam/seed_1000
```

These analysis commands rewrite only the derived `diagnostics.json` and
`diagnostics.png` files; they do not repeat training.
