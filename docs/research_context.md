# Research context

This independent study investigates collocation dynamics in physics-informed
neural networks, with a possible publication as its longer-term objective.
This document records the scientific purpose, study plan, and decisions that
connect individual campaigns. General procedures belong in
[methodology](methodology.md), campaign protocols and interpretation in
[experimental notes](experimental_notes.md), and concise comparisons in
[results](results.md).

## Research questions and working hypotheses

1. Does the relative benefit of residual-guided collocation over sampling
   baselines increase as the solution develops sharper, localized features?
2. Does adding stochastic diffusion to PACMANN improve solution accuracy, and
   does that effect depend on the solution regime?

Our intuition motivates these questions; neither benefit is established.
Sharp solution gradients and coordinate gradients of the residual score are
different quantities. The study must measure solution sharpness independently
of adaptive point placement and evaluate both global and feature-region error.
Residual statistics and point histories help interpret outcomes but cannot
replace solution-error comparisons.

## Stage 1: original-benchmark reproduction and diffusion extension

Use the original PACMANN Burgers reference dataset at viscosity `0.01/pi` and
the released training and sampling settings. The planned ten-seed comparison
includes fixed Hammersley, PACMANN-Adam, and PACMANN with diffusion. First tune
sigma on this same benchmark using separate tuning seeds, then freeze it before
the ten-seed evaluation. Sigma zero remains a valid outcome.

The unmodified methods assess reproduction of the upstream reported results;
the diffusion variant assesses a modification under those benchmark conditions.
Preserve and disclose upstream behavior, including unequal PDE populations and
possible L-BFGS iteration overshoot. This reproduction is distinct from a later
comparison with strictly matched populations and executed budgets.

This stage can establish benchmark-specific diffusion evidence and reproduction
agreement or discrepancies. One viscosity cannot establish how either method's
benefit changes with sharpness. Separate seeds assess repeatability on the same
reference grid, not generalization to unseen reference data or PDE parameters.

The [initial sigma-tuning protocol](experimental_notes.md#burgers-sigma-tuning-01)
is defined, but the comparison is incomplete and no sigma has been selected.
The final ten-seed campaign protocol will follow the tuning decision.

## Stage 2: parameterized sharpness study

Generate matching reference solutions over a parameter range containing smooth,
intermediate, and sharp regimes. Burgers viscosity is the first candidate.
Our professor's proposed high-order MATLAB solver is a potential source; its
availability and numerical accuracy still need to be established. Additional
PDE families require their own validated reference solutions. We do not need
these new datasets to complete Stage 1.

For the main parameterized study, select one sigma per PDE family on a small
development set spanning the intended regimes, then keep it fixed across that
family's final parameter sweep. Give development conditions equal weight using
errors normalized to their matched PACMANN baselines. The exact aggregate score
belongs in the campaign protocol. A benchmark-specific sigma from Stage 1 is a
candidate for this development search, not automatically the final choice.

Use separate tuning and evaluation seeds, and include parameter values excluded
from tuning when claiming transfer across the range. Different PDE families may
select different sigmas. Per-condition tuning may be a follow-up, but it answers
how diffusion performs after local adjustment rather than the main question
about performance as sharpness changes with fixed method settings.

The intended broader comparison includes fixed Hammersley, upstream random
resampling, RAD, PACMANN, and PACMANN with diffusion. Exact resampling settings,
parameter ranges, sharpness measures, feature regions, and training budgets
remain to be specified before that campaign. Examine the change in relative
method benefit across regimes; a method ranking at one parameter value alone
does not answer the sharpness question.

## Decisions and open dependencies

- Keep the two research questions connected through a diffusion ablation and
  comparisons across regimes; a useful result for one does not require a
  positive result for the other.
- Treat tuning as scientific development and the frozen ten-seed comparisons as
  the main evaluation evidence. Engineering readiness checks are separate.
- Preserve upstream settings for reproduction; specify matched controls and a
  fair tuning policy for subsequent method comparisons.
- Resolve reference generation, parameter ranges, sharpness metrics, and the
  final parameterized protocol before making cross-regime claims. Negative or
  uncertain outcomes remain valid results within the tested scope.
