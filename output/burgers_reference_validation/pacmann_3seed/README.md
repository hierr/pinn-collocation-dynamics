# PACMANN errors against alternative Burgers references

Reload the final weights of [three existing PACMANN models](../source_models/)
and evaluate them at PACMANN's original 256 spatial points and 100 times through
`0.99`. Compare those same predictions with the original, N512, and N1024
reference values on that grid. Generated solutions use Chebyshev barycentric
interpolation in space and exact shared snapshot times. Evaluation is CPU
float64; no training occurs.

- [Error table](model_errors.csv): relative L2 fractions per seed and reference;
  multiply by 100 for percentages.
- [Figure](comparison.png): each line is one fixed model; markers change only
  the reference used to measure its overall error. The horizontal axis is
  reference choice, not time or training progress.

N1024 changes the original error scores by less than 0.006% relative; N512
increases them by approximately 22–80%. This supports N1024 for viscosity
`0.01/pi` over `[-1, 1] × [0, 0.99]`. Other viscosities need their own checks.

Reference-to-reference disagreements are calculated separately by
[burguers_compare_solvers.py](../../../scripts/dataset_validation/burguers_compare_solvers.py):
see [N512](../n512_reference_grid/README.md) and
[N1024](../n1024_reference_grid/README.md).

## Execution

[burguers_compare_pinn_solutions.py](../../../scripts/dataset_validation/burguers_compare_pinn_solutions.py)
contains the input paths and seeds. From the repository root:

```bash
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_validation.burguers_compare_pinn_solutions
```

The script regenerates the derived table and figure. Original datasets, weights,
and retained source artifacts remain unchanged. The
[campaign README](../README.md) describes source-model provenance and the shared
comparison procedure.
