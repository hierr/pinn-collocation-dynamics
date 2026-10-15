# N1024 comparison on the original reference grid

The [corrected comparison script](../../../scripts/dataset_validation/burguers_compare_solvers.py)
evaluates the N1024 Chebyshev solution on PACMANN's original 256 spatial points
and 100 exact snapshot times through `0.99`, leaving PACMANN values unchanged.

```bash
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_validation.burguers_compare_solvers
```

[Overall relative L2](overall_error_metrics.txt) is `3.18875102e-6`, or
`0.000318875%`. Select `RESOLUTION = 1024` at the top of the script. The
[difference plot](solution_difference.png) uses the same spatial interpolation
and matching times. The [three-model check](../pacmann_3seed/README.md) provides
the broader validation context.
