# Research context

This independent study investigates collocation dynamics in physics-informed
neural networks, with a possible publication as its longer-term objective.
This document records the scientific purpose, study plan, and decisions that
connect the study phases. Campaign-specific reasoning and execution instructions
belong in campaign READMEs, comparisons and brief interpretation in
[results](results.md), and setup instructions in the [repository README](../README.md).

## Research questions and working hypotheses

1. How does the benefit of point-adaptive collocation over sampling baselines
   change across solution-sharpness regimes, for solution and PDE-residual error?
2. Does adding stochastic diffusion to PACMANN improve solution and PDE-residual
   accuracy, and does that effect depend on the solution regime?

Our intuition motivates these questions; neither benefit is established.
Sharp solution gradients and coordinate gradients of the residual score are
different quantities. The study must measure solution sharpness independently
of adaptive point placement and evaluate both global and feature-region error.
Solution error and independently evaluated PDE-residual error are distinct
outcomes; improvements in one do not imply improvements in the other. Point
histories help interpret these outcomes.

Upstream reproduction serves as an implementation and reference-error check.
The main scientific objective is a controlled comparison across sharpness
regimes. Preserve upstream attribution and disclose changes, without retaining
implementation quirks that undermine matched comparisons.

## Study design principles

Use controlled comparisons with matched settings except for the variables under
study, and disclose deviations needed for upstream reproduction. Pair seeds,
initialization, and initial point designs where applicable. Separate tuning from
evaluation and freeze selected settings before evaluation, with a fair tuning
policy and disclosed costs for competing methods.

Validate reference solutions against the PDE parameters and document numerical
accuracy when generating new data. Evaluate on fixed reference sets independent
of collocation points, with metrics and model-selection rules chosen before
execution. Measure sharpness and feature regions independently of adaptive
sampling. Report repeated-seed variability, unfavorable outcomes, missing runs,
and scientifically consequential deviations. Interpret findings within the
tested problems, parameters, and evaluation conditions.

## Current development and evaluation plan

This persistent branch retains hyperparameter development. Dataset validation
lives on `dataset-validation`; `main` retains accepted code, frozen evaluation
settings, and full evaluation results. Transfer selected code and conclusions
with their source commit; do not merge entire workstreams and their artifacts.

The historical original-reference Burgers sweep selected `1e-4` on three tuning
seeds under its predeclared minimum-mean rule; see the
[campaign design](../output/burgers_sigma_tuning_01_baseline/README.md) and
[results](results.md). This is development evidence for that reference, not a
frozen choice for future generated references or proof of diffusion benefit.

The next phase uses generated Burgers references over a viscosity range spanning
smooth, intermediate, and sharp regimes. The N1024 reference at `0.01/pi` was
accepted after validation on `dataset-validation`; new viscosities require their
own numerical-accuracy checks.

Tune diffusion separately at each viscosity using development seeds, then
freeze settings before ten-seed comparisons of fixed Hammersley, upstream random
resampling, RAD, PACMANN, and PACMANN with diffusion. Per-viscosity tuning measures
performance after local adjustment; it does not establish transfer of one fixed
sigma across regimes. Additional PDE families may follow later.

The historical sweep informs the next search range; a separate re-scoring
campaign is unnecessary. It does not replace tuning under revised training
controls: changing PDE point populations requires new training, whereas changing
only reference values allows existing predictions to be re-scored.

The controlled Burgers implementation loads generated references, matches PDE
populations, and evaluates final solution, residual, and constraint errors.
Additional samplers and sharpness/feature-region evaluation remain future work.

## Accepted controls and evaluation policy

- Match interior PDE point counts from the start and keep identical boundary
  and initial-condition constraints fixed across methods. Remove the released
  implementation's method-dependent inclusion of constraints in the PDE loss.
- Match network architecture, loss weights, and optimizer stages and budgets
  across methods. Start with shared settings across viscosities; any change
  needs a stated experimental rationale. Record actual optimizer work and
  runtime, since equal training schedules do not imply equal computation.
- Select sigma by the lowest mean final-model solution relative L2 over
  development seeds at each viscosity, including zero as a valid candidate.
  Evaluate the same frozen setting for PDE-residual error. Do not choose a
  second sigma for residual error or combine the outcomes into an undeclared
  selection score. Report improvements and tradeoffs separately.
- Evaluate solution error against validated reference values on fixed
  coordinates. Evaluate PDE-residual RMS on a common fixed interior evaluation
  set independent of point adaptation, using derivatives of the trained PINN.
  Training residuals at each method's own collocation points are diagnostics,
  not the residual comparison. Report boundary and initial-condition errors
  separately; do not normalize residual error by the zero exact residual.
- Use ordinary relative L2 and MSE on a common uniform solution grid: initially
  1,024 spatial points including the boundaries, and the 199 native uniformly
  spaced reference times through `0.99`. Interpolate the generated Chebyshev
  reference only in space using barycentric interpolation; infer the PINN
  directly at these coordinates. Count each space-time point equally. Use
  ordinary RMSE for boundary and initial-condition errors, and a fixed uniform
  interior midpoint grid for residual RMS. Check evaluation resolution before
  tuning each sharper regime.
- Define sharpness and feature regions from the reference solution independently
  of adaptive sampling. Compare method benefits across regimes using global and
  feature-region errors, rather than inferring a sharpness effect from a single
  viscosity or method ranking.
- Freeze settings before ten-seed evaluation using seeds excluded from tuning
  and previously inspected readiness runs. Pair seeds across methods and verify
  matching initial weights and point designs. Use final-model errors, retain
  individual outcomes and failures, and report means, sample variability, and
  paired comparisons. Do not retune on the final evaluation results.

## Decisions and open dependencies

- Keep the two research questions connected through a diffusion ablation and
  comparisons across regimes; a useful result for one does not require a
  positive result for the other.
- Treat tuning as scientific development and the frozen ten-seed comparisons as
  the main evaluation evidence. Engineering readiness checks are separate.
- Keep historical reproduction and tuning artifacts under their original
  protocols; apply the controls above to new campaigns.
- At `nu = 0.01/pi`, a six-model evaluation check supported uniform-grid scoring:
  moving from 1,024 to 2,048 spatial points changed relative L2 by less than
  0.003% relative, and ordinary uniform-grid scores differed from native-grid
  quadrature scores by less than 0.25% relative. This supports the simpler
  evaluation rule at this viscosity, not its adequacy for sharper solutions or
  a claim about diffusion effectiveness.
- Specify viscosities and reference-accuracy checks, evaluation coordinates and
  resolution, sharpness and feature-region definitions, point counts, training and
  tuning budgets, search stopping rules, seed lists, and failure reporting in an
  approved campaign protocol before execution. Negative or uncertain outcomes
  remain valid results within the tested scope.

## Supporting references

- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist):
  experimental settings, reproducibility, uncertainty, and compute reporting.
- [Research-code release guidance](https://github.com/paperswithcode/releasing-research-code):
  documented training, evaluation, environments, and saved artifacts.
- [Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html):
  selection bias and the distinction between selection and evaluation.
- [Sivaprasad et al., 2020](https://proceedings.mlr.press/v119/sivaprasad20a.html):
  accounting for hyperparameter tuning effort in method comparisons.
