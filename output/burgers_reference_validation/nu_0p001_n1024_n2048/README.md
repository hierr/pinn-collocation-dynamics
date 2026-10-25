# Burgers spatial-resolution check at ν = 0.001

This check assesses whether spatial discretization could distort the sharper
viscosity study before diffusion tuning. Both saved references use the unchanged
Chebyshev-collocation/Radau solver: 201 snapshots through `t = 1`,
`rtol = 1e-11`, `atol = 1e-13`, and `max_step = 0.005`. Only the polynomial
degree changes, from N1024 (1,025 nodes) to N2048 (2,049 nodes). The N1024 input
is preserved from dataset commit `52b9359`; the N2048 generation entry point was
introduced in `a496b14`.

## Comparison

Use the 199 exact shared snapshots through `t = 0.99`, matching the PINN study.
Compare values directly at the 1,025 nested spatial nodes, then evaluate both
saved spectral polynomials on 8,193 uniform points over `[-1, 1]`. The latter
checks between the shared nodes and avoids the spatial weighting implicit in
an ordinary Euclidean norm over nonuniform Chebyshev nodes. No temporal
interpolation or new solve occurs.

Relative L2 is `||coarse - fine||₂ / ||fine||₂`, pooling the selected space-time
samples; N2048 is the comparison denominator, not an established exact solution.
Spatial gradients are obtained by spectral differentiation, not finite
differences on the uniform comparison grid. The front figure evaluates both
polynomials at 2,001 points in `[-0.025, 0.025]` at `t = 0.35, 0.505, 0.99`.
Spatial interpolation uses [Chebyshev barycentric weights](https://docs.scipy.org/doc/scipy-1.16.0/reference/generated/scipy.interpolate.BarycentricInterpolator.html).

## Findings and decision

| Quantity | Disagreement |
| --- | ---: |
| Solution relative L2, direct shared nodes | 0.0982518% |
| Solution relative L2, dense uniform grid | 0.261572% |
| Maximum absolute solution difference, dense uniform grid | 0.0623938 |
| Spatial-gradient relative L2, dense uniform grid | 14.4540% |
| Maximum relative change in the instantaneous peak gradient, using N2048 as denominator | 15.8760% |

The largest sampled solution difference occurs at `t = 0.505`,
`x = 0.001220703125`. Peak `|u_x|` increases from 410.362 at `t = 0.5` to
487.805 at `t = 0.505`. Early snapshots agree closely; substantial disagreement
appears as the front steepens. The N1024 front profile and derivative show
oscillations that diminish on refinement.

The maximum, across snapshots, of the largest coefficient in the highest 10%
of Chebyshev modes divided by the largest coefficient in the same snapshot
falls from `6.36293e-4` to `3.86673e-5`. This is a spectral-resolution diagnostic,
not a bound on solution error.

N1024 is not accepted for this viscosity. The comparison also does not establish
N2048 accuracy: no asymptotic convergence rate, absolute error bound, or temporal
refinement has been measured. Recommend comparing N2048 against N4096 before
tuning. This check stops at the evidence of
material resolution dependence, without changing the solver or training methods.

## Reproduction and artifacts

From the repository root, using the [documented environment](../../../README.md):

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
MPLBACKEND=Agg venv/bin/python -m scripts.dataset_validation.burgers_compare_resolutions
```

The [script](../../../scripts/dataset_validation/burgers_compare_resolutions.py)
contains all comparison settings and preserves both inputs. `summary.json`
contains pooled metrics, grid settings, peak gradients, and spectral-tail ratios;
`time_series.csv` contains the per-snapshot solution discrepancies and peak
gradients. All relative quantities in these numerical files are fractions;
multiply by 100 for percent. `front_profiles.png` shows the local solution and
gradient; `resolution_over_time.png` shows solution disagreement and peak
gradients over physical time. See [study context](../../../docs/research_context.md),
[results](../../../docs/results.md), and [artifact meanings](../../../docs/output_format.md).
