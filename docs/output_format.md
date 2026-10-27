# Dataset validation artifacts

Reference datasets are under `data/burgers_dataset/`:

| Path | Meaning |
| --- | --- |
| `PACMANN_Burgers.npz` | Original reference: `x` (256, 1), `t` (100, 1), `usol` (256, 100). |
| `burgers_nu_0p0031830989_N{512,1024}.npz` | Generated Chebyshev references: `x` has N+1 nodes, `t` has 201 snapshots through 1.0, `U` is time by space; includes viscosity and solver settings. |
| `burgers_nu_0p001_N{1024,2048,4096}.npz` | Sharper-case resolution inputs with the same field layout. N4096 records `jacobian: analytic` in `config_json`; earlier inputs used numerical Jacobians. |

Retained artifacts are under `output/burgers_reference_validation/`. Its
[campaign README](../output/burgers_reference_validation/README.md) describes
execution and source-model provenance.

| Path | Meaning |
| --- | --- |
| `n{512,1024}_reference_grid/` | Solver comparisons on PACMANN's original coordinates and times; README explains execution. |
| `overall_error_metrics.txt` | Global reference disagreement, including relative L2 as a fraction. |
| `solution_difference.png` | Pointwise difference plots at selected shared times. |
| `solutions_overlay.png`, `space_time_heatmaps.png` | Reference solution views; consult plot axes for each grid. |
| `pacmann_3seed/model_errors.csv` | Nine rows: seed, reference choice, and final relative L2 as a fraction; multiply by 100 for percent. |
| `pacmann_3seed/comparison.png` | One line per fixed model, changing only the reference used to score its predictions on the original grid. |
| `source_models/seed_<seed>/` | Immutable evaluation inputs copied from the original PACMANN baseline. |
| `nu_0p001_n1024_n2048/`, `nu_0p001_n2048_n4096/` | Resolution checks: `summary.json` for pooled disagreement and spectral diagnostics, `time_series.csv` for per-snapshot comparisons, and two figures; each README defines metrics and limitations. Relative quantities are fractions. |

Each source-model directory retains its original `config.yaml` (including
selected seed and network layers), `metadata.json` (training revision, dependency
versions, device, and dataset fingerprint), `models/final.pt` (network state), and
`predictions.npz` (`coordinates`, `reference`, `prediction`). The validation script
reloads weights and checks predictions against the archived arrays before scoring
alternative references. These inputs do not support exact training resume.

Complete original training campaigns remain on `hyperparameter-tuning` and in
Git history. Derived comparisons can be regenerated without training. Original
datasets, saved weights, and provenance must not be overwritten.
