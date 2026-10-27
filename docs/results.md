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

These changes preclude accepting N1024 for this viscosity. Further refinement
gives the following [N2048–N4096 comparison](../output/burgers_reference_validation/nu_0p001_n2048_n4096/README.md):

| Dense-grid quantity | N1024–N2048 | N2048–N4096 |
| --- | ---: | ---: |
| Solution relative L2 disagreement (%) | 0.261572 | 0.00964855 |
| Maximum absolute solution difference | 0.0623938 | 0.00324882 |
| Spatial-gradient relative L2 disagreement (%) | 14.4540 | 1.00069 |

The solution discrepancy decreases by a factor of 27.1, while the peak gradient
changes from 487.805 to 494.986 at the same sampled time `t = 0.505`. N4096 has
a highest-10%-mode coefficient ratio of `1.34791e-7`, compared with `3.86673e-5`
for N2048. The finer front profiles closely overlap.

N4096 is accepted as the working reference for the next development campaign,
with substantially stronger convergence evidence. Pairwise disagreement and spectral-tail ratios
are not absolute-error bounds, and no numerical acceptance threshold was
prescribed. N2048 used numerical Jacobians and N4096 an independently checked
analytical Jacobian of the same RHS; PDE and temporal settings match. This
decision does not certify temporal convergence or absolute accuracy.
Evaluation-grid adequacy remains to be checked before sigma selection.
