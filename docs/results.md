# Experimental results

This document consolidates accepted comparisons and brief interpretation.
Detailed validation and tuning artifacts remain on their respective branches;
full frozen-setting evaluation campaigns are retained on `main`.

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
settings did not imply equal realized computation. The frozen evaluation below
assesses this selection on new seeds.

Detailed comparisons and artifacts remain on `hyperparameter-tuning`, source
commit [`7fffcb6`](https://github.com/hierr/pinn-collocation-dynamics/commit/7fffcb600c62208958d5681d1bcf3cf5a8618148).

## Frozen N1024 ten-seed evaluation

All five methods completed the frozen evaluation on paired seeds 4000–4009 at
`nu = 0.01/pi` using the generated N1024 reference. PACMANN runs used source
revision `9c3f022`; the other four methods used `e02098d`, as recorded in each
run's metadata. All 50 runs completed training and post-processing with finite
predictions. See the
[campaign protocol](../output/burgers/burgers_nu_0p0031830989_10seed/README.md),
[frozen configuration](../config/burgers_nu_0p0031830989_10seed.yaml), and
[per-method artifacts](../output/burgers/burgers_nu_0p0031830989_10seed/).

### Consolidated evaluation results

Primary summaries use every planned seed and report mean ± sample SD.
Solution relative L2 is expressed as percent; residual RMS is unnormalized.

| Method | Completed seeds | Solution L2 (%) | Trimmed solution L2 (%) | Residual RMS | Trimmed residual RMS | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Hammersley | 10/10 | 6.795 ± 20.087 | 0.4786 | 0.44446 ± 1.30908 | 0.03185 | 9420.8 ± 169.6 |
| Random resampling | 10/10 | 0.448 ± 0.333 | 0.4228 | 0.02008 ± 0.00953 | 0.01866 | 9009.7 ± 167.6 |
| RAD | 10/10 | 0.425 ± 1.095 | 0.0829 | 0.01489 ± 0.02724 | 0.00701 | 8928.1 ± 9.2 |
| PACMANN | 10/10 | 11.913 ± 25.053 | 6.9491 | 0.59393 ± 1.24336 | 0.36952 | 9024.4 ± 72.0 |
| PACMANN + diffusion | 10/10 | 12.729 ± 23.518 | 7.1840 | 0.77939 ± 1.55871 | 0.38434 | 9020.2 ± 50.8 |

The supplementary **10% trimmed mean** removes one lowest and one highest value
for each method and error metric independently, then averages the remaining
eight values. It describes sensitivity to extreme outcomes; it does not replace
the full ten-seed summaries or remove seeds from paired comparisons. This
supplementary analysis was specified after inspecting the PACMANN evaluation
outcomes.

For PACMANN, eight seeds have solution L2 errors between **0.03719% and
0.16567%**. Seeds 4002 and 4006 have much larger errors, **55.09497% and
63.50343%**, with residual RMS **2.925526 and 2.980313**, respectively. Their
predictions are finite, and
relative L2 recomputed directly from the saved prediction arrays matches the
recorded metrics. These completed, large-error outcomes remain in the primary
summary. Removing one extreme at each end still retains seed 4002, so the trimmed
means also reflect a large-error outcome.

Other means are also sensitive to large-error outcomes. Fixed Hammersley's
seed 4006 has solution L2 **63.96177%**. Diffusion seeds 4002, 4004, and 4006
have **69.78852%, 21.12525%, and 35.79689%**, respectively. RAD's seed 4006
has **3.53848%**, while its other nine errors are below 0.18%. All these
outcomes are retained.

### Paired method contrasts

Each entry is the paired difference `first method − second method`, reported as
mean ± sample SD over the ten seeds. Solution differences are percentage points;
residual differences are in residual RMS units. Negative differences favor the
first method. Counts are improved / tied / worse for the first method.

| Contrast | Solution L2 difference (pp) | Improved / tied / worse | Residual RMS difference | Improved / tied / worse |
| --- | ---: | ---: | ---: | ---: |
| PACMANN + diffusion − PACMANN | +0.816 ± 12.564 | 4 / 0 / 6 | +0.18546 ± 0.66831 | 4 / 0 / 6 |
| RAD − Fixed Hammersley | −6.370 ± 18.994 | 10 / 0 / 0 | −0.42957 ± 1.28241 | 10 / 0 / 0 |
| PACMANN − Fixed Hammersley | +5.118 ± 17.411 | 9 / 0 / 1 | +0.14947 ± 1.03195 | 9 / 0 / 1 |
| PACMANN + diffusion − Fixed Hammersley | +5.934 ± 25.097 | 8 / 0 / 2 | +0.33493 ± 1.67020 | 8 / 0 / 2 |
| RAD − Random resampling | −0.023 ± 0.978 | 8 / 0 / 2 | −0.00519 ± 0.02690 | 8 / 0 / 2 |
| PACMANN − Random resampling | +11.465 ± 24.998 | 8 / 0 / 2 | +0.57385 ± 1.24374 | 8 / 0 / 2 |
| PACMANN + diffusion − Random resampling | +12.281 ± 23.540 | 6 / 0 / 4 | +0.75931 ± 1.56087 | 7 / 0 / 3 |

On the held-out seeds, the selected diffusion setting did not improve over
PACMANN on average: solution L2 was higher by 0.816 percentage points and
residual RMS by 0.18546, with diffusion lower in four of ten paired runs on each
metric. This does not support a reproducible diffusion benefit at this setting;
the wide paired spread also cautions against treating the mean difference as a
stable effect. RAD had lower errors than fixed Hammersley in all ten paired
runs. Compared with random resampling, RAD improved both metrics in eight of ten
paired runs, but its mean errors were close (solution L2 0.425% vs 0.448%;
residual RMS 0.0149 vs 0.0201), with substantial paired variability.

Mean differences and win counts describe different aspects of performance.
PACMANN improves over fixed Hammersley in nine paired seeds on both metrics,
yet has worse mean errors because the unfavorable outcome outweighs those
improvements. The large-error outcomes must be considered alongside the paired
counts when interpreting method rankings.

### Computation and scope

Every run executed 35,000 network Adam steps. Total L-BFGS iterations and
function evaluations per run are reported below as mean ± sample SD. Actual
work differs despite shared configured limits; the retained DeepXDE driver can
exceed those limits.

| Method | L-BFGS iterations | L-BFGS function evaluations |
| --- | ---: | ---: |
| Fixed Hammersley | 15799.1 ± 473.3 | 21997.6 ± 2282.1 |
| Random resampling | 15696.8 ± 464.7 | 19542.0 ± 2015.9 |
| RAD | 15000.0 ± 0.0 | 16402.3 ± 127.8 |
| PACMANN | 15318.0 ± 397.1 | 17367.1 ± 1224.6 |
| PACMANN + diffusion | 15218.9 ± 363.6 | 17088.7 ± 1268.8 |

Timing reflects up to ten concurrent seeds on one RTX 5070, including competition
with other workers performing final evaluation. Compare it only with method
batches run under the same execution conditions, separately from serial tuning
timings. Diffusion used a development-selected sigma; the other samplers retained
upstream settings. These results describe this viscosity and tuning policy;
they do not establish a sharpness-dependent advantage. The selected diffusion
setting applies only at `nu = 0.01/pi`; other viscosities require their own
reference checks and development tuning. See the
[study plan](research_context.md).
