# Experimental results

This document consolidates accepted comparisons and brief interpretation.
Detailed validation and tuning artifacts remain on their respective branches;
full frozen-setting evaluation campaigns will be retained on `main`.

## Generated Burgers reference validation

At viscosity `0.01/pi`, the generated N1024 solution was validated against the
original PACMANN reference over `[-1, 1] × [0, 0.99]`. Chebyshev barycentric
interpolation onto the original 256-point spatial grid and its 100 exact snapshot
times gave a reference disagreement of **0.000319% relative L2**.

Using the same final PACMANN models for seeds 2000–2002, replacing the original
reference with N1024 on that common grid changed the models' final relative L2
error scores by **less than 0.006% relative to the original scores**. This is a
change in measured model error, distinct from the reference-solution difference.
No retraining was needed. The check supports N1024 for this benchmark; it does
not certify absolute solver accuracy or other viscosities.

Evidence is retained on `dataset-validation`, commit `3a7b653`, in
`dataset_validation/` and `data/burgers_dataset/reference_comparison/`.
The latter includes the comparison READMEs, N512/N1024 solver comparisons,
three-model error table and figure, and unchanged source-model inputs.

## N1024 diffusion development selection

At `nu = 0.01/pi`, the controlled generated-reference sweep compared PACMANN
with diffusion scales `1e-5`, `1e-4`, `1e-3`, and `1e-2` over paired development
seeds 3000–3002. The minimum mean final solution relative L2 selected `sigma=1e-4`:
**0.07963%**, versus **0.14039%** for PACMANN without diffusion. Solution error
improved in two of three seeds. At that same selected sigma, mean independent
residual RMS was **0.003648**, versus **0.004322**, also improving in two seeds.

This is a development selection at one viscosity, not evidence of a general
diffusion benefit or a sharpness effect. DeepXDE's L-BFGS driver sometimes
exceeded the configured stage iteration and function-evaluation limits; shared
settings did not imply equal realized computation. Evaluate the selected
candidate on new seeds.

Detailed comparisons and artifacts remain on `hyperparameter-tuning`, source
commit [`7fffcb6`](https://github.com/hierr/pinn-collocation-dynamics/commit/7fffcb600c62208958d5681d1bcf3cf5a8618148).

## Frozen N1024 ten-seed evaluation

The PACMANN batch completed all ten frozen evaluation seeds, 4000–4009, at
`nu = 0.01/pi` using the generated N1024 reference and source revision `9c3f022`.
All ten runs completed training and post-processing, with finite predictions
and 700 point-movement events per run. See the
[campaign protocol](../output/burgers_nu_0p0031830989_10seed/README.md),
[frozen configuration](../config/burgers_nu_0p0031830989_10seed.yaml), and
[per-seed artifacts](../output/burgers_nu_0p0031830989_10seed/burgers/pacmann_adam/).

### Consolidated evaluation results

Primary summaries use every planned seed and report mean ± sample SD.
Solution relative L2 is expressed as percent; residual RMS is unnormalized.

| Method | Completed seeds | Solution L2 (%) | Trimmed solution L2 (%) | Residual RMS | Trimmed residual RMS | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PACMANN | 10/10 | 11.913 ± 25.053 | 6.9491 | 0.59393 ± 1.24336 | 0.36952 | 9024.4 ± 72.0 |

The supplementary **10% trimmed mean** removes one lowest and one highest value
for each error metric independently, then averages the remaining eight values.
For solution L2, this removes seeds 4007 and 4006; for residual RMS, seeds 4004
and 4006. This descriptive summary was added after observing the first PACMANN
batch. It does not replace the full ten-seed summaries or remove seeds from
paired method comparisons.

Eight seeds have solution L2 errors between **0.03719% and 0.16567%**. Seeds
4002 and 4006 have much larger errors, **55.09497% and 63.50343%**, with residual
RMS **2.925526 and 2.980313**, respectively. Their predictions are finite, and
relative L2 recomputed directly from the saved prediction arrays matches the
recorded metrics. These completed, large-error outcomes remain in the primary
summary. Removing one extreme at each end still retains seed 4002, so the trimmed
means also reflect a large-error outcome.

Timing reflects up to ten concurrent seeds on one RTX 5070, including competition
with other workers performing final evaluation. Compare it with method batches
run under the same execution conditions, separately from serial tuning timings.

Fixed Hammersley, random resampling, RAD, and PACMANN with diffusion remain
pending. This batch alone establishes no method advantage or sharpness effect.
The selected diffusion setting applies only at `nu = 0.01/pi`; other viscosities
require their own reference checks and development tuning. See the
[study plan](research_context.md).
