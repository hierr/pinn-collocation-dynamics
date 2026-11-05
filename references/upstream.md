# Source provenance and implementation choices

## PACMANN source

- Authors: Coen Visser, Alexander Heinlein, and Bianca Giovanardi.
- Paper: *PACMANN: Point adaptive collocation method for artificial neural networks* (2026).
- DOI: https://doi.org/10.1016/j.cma.2025.118723
- Repository: https://github.com/CoenVisser/PACMANN
- Imported commit: `ecf06a6cf277b8beb6f8a99c88baa2535738fd7f`.

The immutable `references/pacmann/burgers/` snapshot contains the released
Burgers scripts, their requirements, the Apache-2.0 license, and an import
manifest recording URLs, byte counts, and content hashes. These scripts are
source evidence; the maintained entry points use `src/` and the project lockfile.
Preserve upstream copyright and license notices. Importing this license does
not assign a project-wide license or establish the dataset's licensing terms.

The retained original is `data/burgers_dataset/PACMANN_Burgers.npz`. The immutable
manifest records its historical local path `data/burgers/Burgers.npz`; its
checksum is unchanged. It has 256 spatial coordinates and 100 saved times.
The upstream example consumes this dataset; the exact historical generator
was not recovered. The generated references used in the controlled study are
separate files and are selected explicitly by configuration.

## Controlled-study choices

| Topic | Maintained behavior |
| --- | --- |
| Domain | x in [-1, 1], t in [0, 0.99], matching the released code |
| Network | Float64; four 64-neuron tanh layers; Glorot normal initialization |
| Network optimization | Five alternating Adam/L-BFGS phases in the scientific campaigns |
| PACMANN cadence | Every 50 network Adam steps; 15 coordinate-Adam ascent steps per event |
| Coordinate Adam | Step size 1e-5; betas (0.9, 0.999); epsilon 1e-7 outside the square root; moments reset per event |
| Escaped points | Uniform replacement after all coordinate steps, following the released timing |
| Constraints | Initial/boundary coordinates stay fixed; only interior PDE points adapt |
| Evaluation | Final checkpoint, rather than selecting training-history minima |
| Seeds | Explicit seed lists and retention of unfavorable outcomes |

The released implementation initially includes 2,740 PDE coordinates: 2,500
interior plus 240 initial/boundary coordinates. Its adaptive callback later
selects the 2,500 interior coordinates. The controlled study matches 2,500
interior PDE coordinates from initialization for every method, alongside the
fixed 80 boundary and 160 initial-condition constraints. This removes a
population mismatch rather than duplicating the released benchmark verbatim.

The code uses coordinate-Adam epsilon 1e-7, while the paper's table reports 1e-8.
PyTorch 2.7.1 and DeepXDE 1.14.0 follow the retained released environment rather
than the paper's stated PyTorch 1.12.1. These choices, generated references,
final-checkpoint scoring, and seed-retention policy mean this submission is a
controlled modified study, not a numerical reproduction of the published table.

Random resampling uses fresh pseudorandom uniform points. RAD uses 100,000
fresh candidates, weights `abs(residual)/mean(abs(residual)) + 1`, and selection
without replacement. Prediction batches limit memory while normalization uses
the whole candidate population. An exactly zero score field is sampled
uniformly rather than dividing by zero.

The diffusion implementation adds independent Gaussian increments with
componentwise standard deviation `sigma * (Lx, Lt) / sqrt(K)` at each coordinate
step. Its persistent NumPy generator is seeded through `SeedSequence([seed, 1])`.
Sigma zero draws no noise. Uniform escaped-point replacement retains the
upstream global random stream. See the [method equations](../docs/peer_review.md#collocation-methods).

## L-BFGS and reference evaluation

The trainer updates DeepXDE's derived per-call limits along with its public
L-BFGS options. Version 1.14.0 otherwise leaves those derived limits at their
import-time defaults. For the full nominal 3,000-iteration stage, calls contain
up to 1,000 iterations. DeepXDE checks the stage limit between calls, so actual
work can exceed the nominal limits. Positive `maxls` enables PyTorch strong-Wolfe
line search; the numerical value 50 is not forwarded as its line-search limit.
Optimizer state resets at each alternating stage, and points stay fixed in
L-BFGS. Actual work and elapsed training time are archived.

Final solution evaluation uses equally weighted uniform spatial coordinates
and exact saved physical times. Generated Chebyshev-Lobatto references are
barycentrically interpolated only in space. Residual evaluation differentiates
the network on an independent midpoint grid and does not use reference values.

The accepted collaborator solver is retained at the
[fixed research checkpoint](https://github.com/hierr/pinn-collocation-dynamics/blob/6e7a638383fe0e04d9d999a2cc491c313f587cc8/scripts/dataset_generation/burgers_spectral.py).
Reference-validation evidence remains on `dataset-validation`, including the
nu = 0.001 refinement checkpoint `41529cb`. Sigma development remains on
`hyperparameter-tuning`; the accepted nu = 0.001 selection is checkpoint `56ea078`.
The review guide summarizes the checks and their limitations.

## Related methods and software

- C. Wu, M. Zhu, Q. Tan, Y. Kartha, and L. Lu. *A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks* (2023). [Reference implementation](https://github.com/lu-group/pinn-sampling).
- L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis. *DeepXDE: A deep learning library for solving differential equations*, SIAM Review (2021). [Project](https://github.com/lululxvi/deepxde).
