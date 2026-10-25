# Dataset validation results

## Generated Burgers reference validation

At viscosity `0.01/pi`, compare generated references with PACMANN on its original
256 spatial points and 100 times over `[-1, 1] × [0, 0.99]`. Use Chebyshev
barycentric interpolation in space and exact shared snapshot times.

| Generated reference | Relative L2 disagreement (%) | Change in fixed-model error scores |
| --- | ---: | --- |
| [N512](../output/burgers_reference_validation/n512_reference_grid/README.md) | 0.0664354 | Approximately 22–80% higher |
| [N1024](../output/burgers_reference_validation/n1024_reference_grid/README.md) | 0.000318875 | Less than 0.006% relative change |

The [model comparison](../output/burgers_reference_validation/pacmann_3seed/README.md)
uses three existing final PACMANN models, seeds 2000–2002. Each model's predictions
are identical across reference choices. Error-score changes are relative to its
original error score; they are neither reference-solution differences nor
percentage-point changes. No training occurs during this check.

These results support N1024 for this benchmark. Agreement with an existing
reference is not an independent proof of absolute accuracy. Other viscosities
and evaluation intervals require their own checks.

## Spatial refinement at ν = 0.001

The [N1024–N2048 comparison](../output/burgers_reference_validation/nu_0p001_n1024_n2048/README.md)
holds temporal settings fixed and compares 199 shared snapshots through `t = 0.99`.
Solution relative L2 disagreement is 0.0982518% at the nested spatial nodes and
0.261572% when both spectral solutions are evaluated on 8,193 uniform points.
The dense-grid maximum absolute difference is 0.0623938 near the front at
`t = 0.505`; peak spatial gradient rises from 410.362 to 487.805 on refinement.
N1024 also shows oscillations near the steep front. Spatial-gradient relative
L2 disagreement on the dense grid is 14.4540%.

These changes preclude accepting N1024 for this viscosity and motivate an
N2048–N4096 comparison before tuning. N2048 remains a candidate rather than a
validated reference; this comparison does not establish absolute accuracy or
temporal convergence.
