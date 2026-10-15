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
