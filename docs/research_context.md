# Research context

This independent study investigates collocation dynamics in physics-informed
neural networks, with a possible publication as its longer-term objective.
This document records the scientific purpose, study plan, and decisions that
connect the study phases. Campaign-specific reasoning and execution instructions
belong in campaign READMEs, comparisons and brief interpretation in
[results](results.md), and setup instructions in the [repository README](../README.md).

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

Parameter values, tuning grids and selection rules, seeds, training budgets,
point populations, sharpness metrics, and feature regions require a concrete
approved protocol before execution. Training currently loads the original
PACMANN reference; configurable generated-reference loading and the additional
samplers remain implementation work.

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

## Supporting references

- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist):
  experimental settings, reproducibility, uncertainty, and compute reporting.
- [Research-code release guidance](https://github.com/paperswithcode/releasing-research-code):
  documented training, evaluation, environments, and saved artifacts.
- [Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html):
  selection bias and the distinction between selection and evaluation.
- [Sivaprasad et al., 2020](https://proceedings.mlr.press/v119/sivaprasad20a.html):
  accounting for hyperparameter tuning effort in method comparisons.
