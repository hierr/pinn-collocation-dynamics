# N512 comparison on the original reference grid

Generated with `RESOLUTION = 512` in the
[Burgers comparison script](../../../../scripts/dataset_generation/validation/burguers_compare_solvers.py):

```bash
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_generation.validation.burguers_compare_solvers
```

The generated solution is interpolated onto PACMANN's unchanged 256 spatial
points at its 100 exact snapshot times through `0.99`.
[Overall relative L2](overall_error_metrics.txt) is `6.64353727e-4`, or
`0.0664354%`. See the [difference plot](solution_difference.png) and
[three-model check](../pacmann_3seed/README.md) for context.
