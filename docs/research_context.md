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

## Research workstreams

Use three persistent branches, retaining each workstream's history:

- `dataset-validation`: reference generation, validation scripts, and evidence.
- `hyperparameter-tuning`: development campaigns, configurations, and selections.
- `main`: accepted implementations and conclusions, frozen evaluation settings,
  and full evaluation results.

Transfer selected code, configurations, or conclusions with the source commit.
Avoid whole-branch merges that import unrelated campaigns. Branch separation
organizes the repository; independent tuning and evaluation seeds and frozen
settings provide scientific separation.

## Reference validation and current evidence

The collaborator's Chebyshev-collocation/Radau Burgers solver provides generated
references. At viscosity `0.01/pi`, N1024 was checked against PACMANN on its
original grid and accepted for this benchmark; see the
[validation result](results.md#generated-burgers-reference-validation).
Other viscosities require their own numerical-accuracy checks.

The earlier pilot remains in Git history; the original-reference sigma sweep
is retained on `hyperparameter-tuning`. They informed development; they are not
the main frozen-setting evaluation. The historical `1e-4` selection does not
freeze sigma for the new generated-reference study. No full ten-seed evaluation
has been completed.

## Parameterized sharpness study

Start with Burgers references over a viscosity range spanning smooth,
intermediate, and sharp regimes. Use the generated references after checking
numerical accuracy. Tune diffusion separately at each viscosity with development
seeds, then freeze settings before ten-seed comparisons of fixed Hammersley,
upstream random resampling, RAD, PACMANN, and PACMANN with diffusion. Other PDE
families can follow later.

This design tests performance after local tuning at each viscosity. It does not
test transfer of one fixed sigma across the range. Disclose tuning costs and
apply a fair tuning policy for competing methods. A ranking at one viscosity
alone does not answer how relative benefit changes with sharpness.

Parameter values, sharpness measures, feature regions, tuning grids and selection
rules, evaluation seeds, point populations, and training budgets remain to be
specified in an approved campaign protocol. Preserve released-code behavior
when assessing reproduction, including its unequal PDE populations and possible
L-BFGS overshoot; specify matched controls for the broader method comparison.

The trainer currently loads the original PACMANN reference and implements fixed
Hammersley, PACMANN-Adam, and PACMANN with diffusion. Configurable generated-data
loading and the other two samplers remain implementation work. The upstream
settings template is not an approved evaluation configuration.

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
