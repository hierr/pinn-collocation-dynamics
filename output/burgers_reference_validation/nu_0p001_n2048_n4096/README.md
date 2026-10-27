# Burgers N2048–N4096 comparison at ν = 0.001

Compare the saved references over `[-1, 1] × [0, 0.99]`, using 199 exact shared
snapshots. N is the Chebyshev polynomial degree, giving N+1 spatial nodes.
Compare the 2,049 nested nodes directly, then evaluate both spectral solutions
and their spatial derivatives on 8,193 uniform points. Relative L2 is
`||N2048 - N4096||₂ / ||N4096||₂`, pooling all selected space-time samples.
N4096 supplies the denominator; it is not an exact solution.

Both inputs use the same PDE, initial/boundary conditions, Chebyshev spatial
operators, Radau integration, `rtol = 1e-11`, `atol = 1e-13`, and
`max_step = 0.005`. N2048 used finite-difference Jacobians; N4096 uses the
analytical derivative of the same interior RHS. This implementation difference
is permitted explicitly and recorded in `summary.json`; the
[Jacobian checks](../../../tests/test_burgers_generation.py) cover its derivative
and short-integration agreement with finite differences. No reference is
regenerated during comparison.

## Results and interpretation

| Quantity | N2048–N4096 disagreement |
| --- | ---: |
| Solution relative L2, direct shared nodes | 0.00156780% |
| Solution relative L2, dense uniform grid | 0.00964855% |
| Maximum absolute solution difference, dense uniform grid | 0.00324882 |
| Spatial-gradient relative L2, dense uniform grid | 1.00069% |
| Maximum relative change in instantaneous peak gradient, N4096 denominator | 1.45062% |

The largest sampled solution difference is near the front at `t = 0.505`,
`x = 0.000732421875`. Peak `|u_x|` changes from 487.805 to 494.986; both peaks
occur at `t = 0.505`. The front solution and gradient profiles closely overlap.

Compared with the [N1024–N2048 check](../nu_0p001_n1024_n2048/README.md), dense-grid
solution disagreement is 27.1 times smaller and gradient disagreement is 14.4
times smaller. The highest-10%-mode coefficient ratio defined there falls from
`3.86673e-5` at N2048 to `1.34791e-7` at N4096.

These observations provide substantially stronger spatial-convergence evidence
and support accepting N4096 as the working reference for development. The
differences are not error bounds for N4096, and spectral-tail decay does not certify absolute
accuracy. No numerical acceptance threshold was prescribed for this comparison;
PINN evaluation-grid adequacy remains to be checked before sigma selection.

## Reproduction and artifacts

Select `DEGREES = (2048, 4096)` in
[`burgers_compare_resolutions.py`](../../../scripts/dataset_validation/burgers_compare_resolutions.py),
then run from the repository root in the [documented environment](../../../README.md):

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_validation.burgers_compare_resolutions
```

`summary.json` contains pooled metrics and input/Jacobian provenance;
`time_series.csv` contains per-snapshot disagreement and peak gradients.
Relative quantities in these numerical files are fractions, not percentages.
`front_profiles.png` shows solutions and gradients at `t = 0.35, 0.505, 0.99`
on 2,001 points in `[-0.025, 0.025]`; `resolution_over_time.png` shows solution
disagreement and peak gradients over time. See [results](../../../docs/results.md),
[study context](../../../docs/research_context.md), and
[artifact meanings](../../../docs/output_format.md).
