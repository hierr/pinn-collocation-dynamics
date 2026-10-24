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

Exploratory inspection of saved optimizer-stage checkpoints showed that some
large-error outcomes followed earlier accurate solutions. PACMANN seed 4002
deteriorated from 0.209% relative L2 after phase 2 to 55.095% finally; diffusion
seed 4004 deteriorated from 0.129% to 21.125%. Their predictions developed
displaced steep fronts, accompanied by large discrepancies between sampled and
independent PDE residuals. Fixed Hammersley seed 4006 was already inaccurate at
the first saved stage. These observations identify distinct failure trajectories
but do not establish their causes.

See the exploratory transition figures for
[PACMANN seed 4002](../output/burgers/burgers_nu_0p0031830989_10seed/diagnostics/pacmann_adam_4002_transition.png)
and
[diffusion seed 4004](../output/burgers/burgers_nu_0p0031830989_10seed/diagnostics/pacmann_diffusion_4004_transition.png),
with [reproduction instructions](../output/burgers/burgers_nu_0p0031830989_10seed/README.md#exploratory-checkpoint-diagnostics).

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

## Diffusion development selection at nu 0.01

The separate sweep at **ν = 0.01** completed all 15 planned runs: ordinary
PACMANN and positive sigmas `1e-5`, `1e-4`, `1e-3`, and `1e-2`, each on paired
development seeds 3000–3002. No numerical failures occurred. Select **sigma =
1e-3** by the lowest mean final solution relative L2 among positive candidates;
this interior winner requires no search expansion. Ordinary PACMANN is the
comparison baseline only. Residual error uses that same solution-selected sigma.

| Setting | Solution relative L2, mean ± sample SD (%) | PDE-residual RMS, mean ± sample SD |
| --- | ---: | ---: |
| PACMANN | 0.0289837 ± 0.0112685 | 0.00187201 ± 0.0000839024 |
| PACMANN + diffusion, sigma 1e-3 | 0.0262913 ± 0.00596281 | 0.00173037 ± 0.000505739 |

On these development seeds, mean solution error is 9.29% lower and mean residual
RMS is 7.57% lower, each improving in two of three paired seeds. These are
selection-set observations, not evidence of a held-out benefit or a sharpness
effect. The generated reference passed structural checks without independent
accuracy certification at this viscosity. The completed ten-seed evaluation is
reported below.

Accepted source: `hyperparameter-tuning` commit `2baa9dd`. Full per-seed results,
the shared protocol, and execution conditions remain in that commit's
`output/burgers/burgers_nu_0p01_sigma_0/README.md`, with all five candidate
campaigns retained on the tuning branch. See the
[study context](research_context.md#reference-validation-and-current-evidence).

## Frozen evaluation at nu 0.01

The frozen evaluation completed all five methods on paired seeds 4000–4009,
using the generated N1024 reference at viscosity `nu = 0.01`. The three
sampling baselines record source revision `c615f04`; PACMANN-Adam and diffusion
record `6582fa7`. Training and scoring source and frozen configuration were
unchanged between those revisions. All 50 runs completed with finite
predictions and no numerical failures. Independent CPU evaluation reproduced
the saved error metrics for all runs, including all 20 PACMANN checkpoints
completed in this phase. See the
[campaign protocol and per-seed results](../output/burgers/burgers_nu_0p01_10seed/README.md)
and [configuration](../config/burgers_nu_0p01_10seed.yaml).

Primary summaries use all ten seeds and report mean ± sample SD. Solution
relative L2 is expressed as a percentage; residual RMS is unnormalized.
Supplementary trimmed means remove one minimum and one maximum separately for
each metric, averaging the remaining eight values. Paired comparisons retain
all ten seeds.

| Method | Completed seeds | Solution L2 (%) | Trimmed solution L2 (%) | Residual RMS | Trimmed residual RMS | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Hammersley | 10/10 | 0.0115567 ± 0.00282369 | 0.0115932 | 0.000916508 ± 0.000128360 | 0.000904519 | 9945.9 ± 127.4 |
| Random resampling | 10/10 | 0.0194975 ± 0.00456948 | 0.0183627 | 0.00120614 ± 0.000175615 | 0.00121857 | 10006.1 ± 112.7 |
| RAD | 10/10 | 0.0172366 ± 0.00431413 | 0.0169540 | 0.000890220 ± 0.0000939736 | 0.000876878 | 9355.7 ± 125.7 |
| PACMANN without diffusion | 10/10 | 0.0324404 ± 0.0130586 | 0.0308782 | 0.00180119 ± 0.000268183 | 0.00175898 | 10170.3 ± 74.3 |
| PACMANN with diffusion, sigma 1e-3 | 10/10 | 0.0257280 ± 0.00785700 | 0.0254398 | 0.00157095 ± 0.000398408 | 0.00157040 | 10278.8 ± 81.1 |

For PACMANN with diffusion minus PACMANN without diffusion, the mean paired
solution L2 difference was **-0.006712 percentage points ± 0.016198**; diffusion
had lower solution error in seven of ten seeds. The mean paired residual RMS
difference was **-0.00023024 ± 0.00037751**, lower in nine of ten seeds.
Negative differences favor diffusion. These are observations for the selected
setting and one viscosity, not a general or sharpness-dependent benefit.

Every run performed 35,000 Adam iterations. Actual L-BFGS work and function
evaluations vary under the retained DeepXDE driver, which can exceed configured
stage limits.

| Method | L-BFGS iterations per seed, mean ± sample SD | L-BFGS function evaluations per seed, mean ± sample SD |
| --- | ---: | ---: |
| Fixed Hammersley | 16322.7 ± 348.6 | 25513.6 ± 1698.4 |
| Random resampling | 16603.6 ± 342.7 | 25799.7 ± 1379.3 |
| RAD | 15735.5 ± 506.9 | 19807.0 ± 1944.8 |
| PACMANN without diffusion | 16344.8 ± 276.0 | 24786.2 ± 904.8 |
| PACMANN with diffusion | 16325.0 ± 476.6 | 25666.4 ± 882.2 |

Training times reflect ten concurrent seeds sharing one RTX 5070; compilation
is included and final evaluation is excluded. These timings are not comparable
to the three-way-concurrent development runs. The three baseline samplers retain
upstream settings while the positive diffusion scale was selected on separate
development seeds. The generated reference passed structural and loader checks,
without independent numerical-accuracy certification at this viscosity. These
results do not resolve sharpness-dependent effects; the selected sigma is
specific to this viscosity and protocol. See the
[study plan](research_context.md).

## Diffusion development selection at nu 0.1

The [nu = 0.1 tuning sweep](https://github.com/hierr/pinn-collocation-dynamics/commit/1968b350a5d44adc9180782406357c2c8763be92)
completed 21 runs on development seeds 3000–3002, including one approved
lower-boundary expansion. It selected positive `sigma = 1e-5` by the lowest
mean final solution relative L2; sigma zero was a comparison baseline only.
The selected positive value is interior among tested candidates. Mean solution
relative L2 was 0.00961106 ± 0.00186134%, compared with 0.0177093 ± 0.00715926%
for ordinary PACMANN. The paired difference was -0.00809823 ± 0.00529978
percentage points, lower in all three seeds. At the same selected sigma, mean
residual RMS was 0.000773829 ± 0.000180949, compared with 0.00115176 ±
0.000400647 for the baseline; the paired difference was -0.000377936 ±
0.000237073, lower in all three seeds. These three-seed development results do
not establish held-out performance or a general diffusion benefit.

The archived training used source commit
[`0948ad9`](https://github.com/hierr/pinn-collocation-dynamics/commit/0948ad936fb32502c19f0b174572866799ca1285).
The frozen five-method evaluation uses seeds 4000–4009 and the separate N1024
reference; see its [configuration](../config/burgers_nu_0p1_10seed.yaml) and
[campaign protocol](../output/burgers/burgers_nu_0p1_10seed/README.md). The
reference passed structural and loader checks but has no independent numerical-
accuracy certification at this viscosity.

## Frozen evaluation at nu 0.1

The [frozen five-method evaluation](../output/burgers/burgers_nu_0p1_10seed/README.md)
completed all 50 runs on paired seeds 4000–4009, using the generated N1024
reference at viscosity **nu = 0.1** and the development-selected **sigma =
1e-5**. Every run records source revision `e261b37`. The reference passed
structural and loader checks, but it has no independent numerical-accuracy
certification at this viscosity. All runs completed with finite saved outcomes
and no numerical failures. Independent CPU checkpoint evaluation reproduced
the five saved error metrics for every run.

Primary summaries use all ten seeds and report mean ± sample SD. Solution
relative L2 is expressed as percent, residual RMS is unnormalized, and training
time includes compilation and optimizer work but excludes final evaluation.
The supplementary 10% trimmed means remove the minimum and maximum independently
for each error metric and average the remaining eight seeds.

| Method | Completed seeds | Solution L2 (%) | Trimmed solution L2 (%) | Residual RMS | Trimmed residual RMS | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Hammersley | 10/10 | 0.0141698 ± 0.0036669 | 0.0140509 | 0.000644720 ± 0.000161940 | 0.000622642 | 9741.4 ± 106.3 |
| Random resampling | 10/10 | 0.0124963 ± 0.0035691 | 0.0124078 | 0.000655814 ± 0.000100990 | 0.000647543 | 9876.2 ± 127.7 |
| RAD | 10/10 | 0.0135256 ± 0.0031563 | 0.0135410 | 0.000541087 ± 0.000065500 | 0.000533121 | 10025.0 ± 189.7 |
| PACMANN | 10/10 | 0.0158011 ± 0.0026410 | 0.0158341 | 0.000906068 ± 0.000145750 | 0.000885028 | 10134.9 ± 132.5 |
| PACMANN + diffusion, sigma 1e-5 | 10/10 | 0.0146387 ± 0.0043371 | 0.0142006 | 0.000806642 ± 0.000118780 | 0.000796086 | 10036.5 ± 344.0 |

Each paired contrast is the first method minus the second over all ten seeds.
Solution differences are percentage points; residual differences retain
residual RMS units. Negative differences favor the first method. Counts are
improved / tied / worse for that first method.

| Contrast | Solution difference (pp), mean ± sample SD | Improved / tied / worse | Residual difference, mean ± sample SD | Improved / tied / worse |
| --- | ---: | ---: | ---: | ---: |
| PACMANN + diffusion − PACMANN | -0.00116244 ± 0.00372667 | 7 / 0 / 3 | -0.0000994261 ± 0.0001731714 | 7 / 0 / 3 |
| RAD − Fixed Hammersley | -0.00064418 ± 0.00376597 | 6 / 0 / 4 | -0.0001036325 ± 0.0001927985 | 7 / 0 / 3 |
| PACMANN − Fixed Hammersley | +0.00163130 ± 0.00538301 | 4 / 0 / 6 | +0.0002613481 ± 0.0001941871 | 1 / 0 / 9 |
| PACMANN + diffusion − Fixed Hammersley | +0.00046886 ± 0.00585162 | 5 / 0 / 5 | +0.0001619220 ± 0.0002387009 | 2 / 0 / 8 |
| RAD − Random resampling | +0.00102937 ± 0.00409004 | 4 / 0 / 6 | -0.0001147264 ± 0.0001145805 | 9 / 0 / 1 |
| PACMANN − Random resampling | +0.00330485 ± 0.00448874 | 2 / 0 / 8 | +0.0002502542 ± 0.0001791322 | 1 / 0 / 9 |
| PACMANN + diffusion − Random resampling | +0.00214241 ± 0.00584420 | 6 / 0 / 4 | +0.0001508281 ± 0.0001553290 | 1 / 0 / 9 |

On these held-out seeds, the selected diffusion setting had lower mean solution
and residual errors than PACMANN without diffusion, with improvements in seven
of ten pairs for both metrics. The paired spreads are wider than the mean
differences, so this comparison does not establish a general diffusion benefit.
Random resampling had the lowest mean solution error, while RAD had the lowest
mean residual RMS. Compared with random resampling, RAD had a slightly higher
mean solution error and lower residual RMS, with the latter lower in nine of
ten paired runs.

Every run performed 35,000 Adam steps. Mean ± sample SD observed L-BFGS work
and function evaluations per seed were:

| Method | L-BFGS iterations | L-BFGS function evaluations |
| --- | ---: | ---: |
| Fixed Hammersley | 16713.6 ± 721.2 | 24811.4 ± 2490.6 |
| Random resampling | 16363.6 ± 720.8 | 25333.2 ± 1514.8 |
| RAD | 16737.6 ± 660.3 | 24161.7 ± 2714.4 |
| PACMANN | 16557.8 ± 481.8 | 25087.1 ± 2712.2 |
| PACMANN + diffusion | 17181.9 ± 494.0 | 25184.1 ± 1485.6 |

Training times reflect ten concurrent seeds sharing one RTX 5070 using
float64, with one numerical CPU thread per worker. Method batches ran
sequentially. These timings should be compared only with runs under the same
concurrency conditions. The result is specific to this generated reference,
viscosity, and per-viscosity tuning policy; it does not isolate a sharpness
effect. See the [study context](research_context.md).

## Diffusion selection at pi/100

The [π/100 tuning sweep](https://github.com/hierr/pinn-collocation-dynamics/commit/e089cb9d02def773fa16b9811e2f2f6ee5c59a7c)
completed all 15 initial-grid runs and selected **sigma = 1e-4**, the positive
candidate with the lowest mean final solution relative L2. This value is
interior among the four tested positive settings, so no expansion was required.
Sigma zero was a comparison baseline only, although its mean solution error
(0.0135490%) was lower than the selected positive value (0.0142509%). The
paired solution difference was +0.000701916 percentage points, with lower
error in one of three seeds. At sigma 1e-4, mean residual RMS was 0.000968064
versus 0.000959376 at sigma zero; the paired residual difference was
+0.00000868754, with lower residual in one of three seeds. Residual did not
select a separate value. These three-seed development outcomes do not establish
held-out performance or a general diffusion benefit.

The tuning artifacts record training source revision
[`cb6c6e8`](https://github.com/hierr/pinn-collocation-dynamics/commit/cb6c6e85642b82a70ce9e64013b74cd5633788ff).
The selected setting is frozen for the separate ten-seed evaluation on seeds
4000–4009; see its [configuration](../config/burgers_nu_0p031415927_10seed.yaml)
and [campaign protocol](../output/burgers/burgers_nu_0p031415927_10seed/README.md).
The restored reference passed structural and loader checks but has no
independent numerical-accuracy certification at this viscosity.

## Frozen evaluation at pi/100

The [five-method evaluation](../output/burgers/burgers_nu_0p031415927_10seed/README.md)
completed all 50 planned runs on paired seeds 4000–4009 at **nu = pi/100**
(approximately 0.031416), with frozen **sigma = 1e-4**. All runs record
source revision `09b380e`, the same pinned environment, CUDA execution, and
the configured N1024 reference fingerprint. Saved outcomes are finite, with
no recorded numerical failures. Independent CPU evaluation of every final
checkpoint reproduced the five saved error metrics. Initial point designs
match across methods, constraints stay fixed, and physics points do not move
during L-BFGS. The reference has no independent numerical-accuracy certification
at this viscosity.

Primary summaries use all ten seeds and report mean ± sample SD. Solution L2
is percent; residual RMS is unnormalized. Supplementary 10% trimmed means
remove one minimum and one maximum independently for each error metric.

| Method | Completed seeds | Solution L2 (%) | Trimmed solution L2 (%) | Residual RMS | Trimmed residual RMS | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Hammersley | 10/10 | 0.0096267 ± 0.0027322 | 0.0096418 | 0.000676303 ± 0.000093294 | 0.000676085 | 10280.5 ± 62.8 |
| Random resampling | 10/10 | 0.0115675 ± 0.0034111 | 0.0115390 | 0.000767601 ± 0.000143477 | 0.000758348 | 10124.7 ± 56.6 |
| RAD | 10/10 | 0.0115145 ± 0.0018647 | 0.0115063 | 0.000682291 ± 0.000117920 | 0.000693719 | 10364.9 ± 217.5 |
| PACMANN | 10/10 | 0.0167485 ± 0.0047115 | 0.0161800 | 0.001114322 ± 0.000202999 | 0.001106621 | 10623.0 ± 121.3 |
| PACMANN + diffusion, sigma 1e-4 | 10/10 | 0.0158724 ± 0.0044469 | 0.0156874 | 0.001132372 ± 0.000241172 | 0.001104242 | 10617.1 ± 102.8 |

Paired differences are the first method minus the second over the same ten
seeds. Solution differences are percentage points; residual differences are
in residual RMS units. Negative differences favor the first method. Counts
are improved / tied / worse; no seeds are removed from these comparisons.

| Contrast | Solution difference (pp), mean ± sample SD | Improved / tied / worse | Residual difference, mean ± sample SD | Improved / tied / worse |
| --- | ---: | ---: | ---: | ---: |
| PACMANN + diffusion − PACMANN | -0.00087609 ± 0.00280704 | 7 / 0 / 3 | +0.0000180498 ± 0.0002006116 | 5 / 0 / 5 |
| RAD − Fixed Hammersley | +0.00188772 ± 0.00346704 | 2 / 0 / 8 | +0.0000059875 ± 0.0001822173 | 5 / 0 / 5 |
| PACMANN − Fixed Hammersley | +0.00712177 ± 0.00534495 | 1 / 0 / 9 | +0.0004380187 ± 0.0002499726 | 0 / 0 / 10 |
| PACMANN + diffusion − Fixed Hammersley | +0.00624568 ± 0.00520237 | 0 / 0 / 10 | +0.0004560685 ± 0.0002412814 | 0 / 0 / 10 |
| RAD − Random resampling | -0.00005308 ± 0.00286948 | 4 / 0 / 6 | -0.0000853101 ± 0.0001164004 | 8 / 0 / 2 |
| PACMANN − Random resampling | +0.00518096 ± 0.00536231 | 1 / 0 / 9 | +0.0003467212 ± 0.0002214248 | 0 / 0 / 10 |
| PACMANN + diffusion − Random resampling | +0.00430488 ± 0.00606849 | 2 / 0 / 8 | +0.0003647710 ± 0.0002776425 | 1 / 0 / 9 |

Fixed Hammersley has the lowest mean solution error and residual RMS. RAD's
mean residual is close to fixed Hammersley's and lower than random resampling's,
with lower residual in eight of ten pairs against random resampling. Both
PACMANN variants have higher residual RMS than fixed Hammersley in every seed.
All final solution errors lie between 0.00537% and 0.02644%.

Diffusion lowers PACMANN's mean solution L2 from 0.0167485% to 0.0158724%,
with lower error in seven of ten pairs. Mean residual RMS instead rises from
0.00111432 to 0.00113237, with lower residual in five of ten pairs. The paired
spread exceeds each mean difference. The tested seed batch shows a modest mean
solution-error reduction, without a consistent improvement across both metrics
or an advantage over fixed Hammersley.

Every run used 35,000 Adam steps. Observed L-BFGS work per seed is mean ± sample
SD; actual work varies despite shared configured limits.

| Method | L-BFGS iterations | L-BFGS function evaluations |
| --- | ---: | ---: |
| Fixed Hammersley | 16878.8 ± 350.3 | 27223.0 ± 1997.6 |
| Random resampling | 16888.1 ± 422.0 | 26128.7 ± 2015.8 |
| RAD | 16774.5 ± 735.4 | 26630.4 ± 3128.2 |
| PACMANN | 17223.9 ± 391.2 | 27478.0 ± 1613.3 |
| PACMANN + diffusion | 16971.7 ± 465.9 | 27752.0 ± 1186.1 |

Training times reflect ten concurrent seeds sharing one RTX 5070 (12 GB),
float64, and one numerical CPU thread per worker. Method batches run
sequentially; timings include compilation and training and exclude final
evaluation. These results concern a generated reference at one viscosity after
per-viscosity diffusion tuning. They do not isolate sharpness as the cause of
method differences. The accepted three-regime figures below retain their
original scope and input snapshot; this campaign is reported separately here.

## Comparison across three Burgers regimes

This post hoc comparison uses all 150 final models from the completed campaigns
at `nu = 0.1`, `0.01`, and `0.01/pi`. The
[analysis script](../scripts/analysis/burgers_regime_comparison.py), transferred
from `analysis` commit `2f8af94`, reads the fixed campaign snapshot `741c19b`.
Solution scores recomputed from saved predictions match the archived final
scores; residual RMS values are the archived independent-grid evaluations.
No new training or inference is performed. The intermediate `pi/100` case is
outside this three-regime comparison.

![Final solution and residual errors for every planned seed](figures/burgers_regime_comparison/final_errors.png)

Full-seed arithmetic means remain the primary summaries. Random resampling has
the lowest mean solution error at `nu=0.1`, fixed Hammersley at `nu=0.01`, and
RAD at `nu=0.01/pi`; RAD has the lowest mean residual RMS in all three cases.
At the sharpest case, PACMANN has the lowest median solution error (0.05659%,
compared with RAD's 0.06642%), but large errors lift its mean to 11.913%.
The distinction between typical accuracy and unfavorable outcomes is retained
rather than resolved by discarding seeds.

The [paired baseline comparisons](figures/burgers_regime_comparison/baseline_ratios.png)
divide each method's error by its baseline's error for the same seed and
viscosity. They show lower median ratios for RAD and PACMANN in the sharpest
case, with occasional large unfavorable ratios. The advantage is not consistent
across the other two viscosities. Connecting median lines are descriptive;
viscosity cases are categorical and are not pooled as independent replications.
Labels use the reference-only peak gradient `G` defined in
[regime characterization](research_context.md#reference-characterization-and-regime-coverage).

The [diffusion comparison](figures/burgers_regime_comparison/diffusion_ratios.png)
isolates diffusion versus ordinary PACMANN at the frozen per-viscosity sigma.
These are medians of same-seed ratios, not ratios of method medians:

| Viscosity | Median solution ratio | Median residual ratio | Lower solution / residual errors |
| --- | ---: | ---: | --- |
| 0.1 | 0.985 | 0.913 | 7/10; 7/10 |
| 0.01 | 0.869 | 0.886 | 7/10; 9/10 |
| 0.01/pi | 1.036 | 1.277 | 4/10; 4/10 |

Diffusion lowers both mean errors relative to PACMANN in the smoother two cases,
but raises both at the sharpest case. This does not establish a general benefit
or make diffusion the method with the lowest mean error in the smoother cases.

### Exploratory temporal evidence in the steepest regime

The [physical-time profiles](figures/burgers_regime_comparison/solution_error_over_time.png)
show mean and median solution errors of the final models over the PDE time
coordinate. At `nu = 0.01/pi`, PACMANN's typical accuracy improves relative to
the sampling baselines as the solution develops a steep front. Selected times
illustrate the pattern; they were inspected after observing the curves and do
not define a pre-specified evaluation window:

| Physical time | PACMANN median solution L2 (%) | Fixed Hammersley median solution L2 (%) | PACMANN lower than fixed Hammersley |
| --- | ---: | ---: | ---: |
| 0.25 | 0.0461 | 0.0281 | 3/10 seeds |
| 0.75 | 0.0705 | 0.7718 | 9/10 seeds |
| 0.99 | 0.0970 | 0.8867 | 9/10 seeds |

Against random resampling, PACMANN has lower solution error in seven of ten
paired seeds at `t = 0.75` and eight at `t = 0.99`. Reference maximum spatial
gradients at `t = 0.25`, `0.75`, and `0.99` are approximately 12.8, 119.1,
and 85.1. The gradient peaks at approximately 151.9 near `t = 0.51`, then
decreases while the front remains steep. Later times therefore represent a
developed steep-front regime, rather than continuously increasing sharpness.

This is exploratory evidence consistent with a regime-dependent benefit in
typical PACMANN accuracy. Robustness remains a separate finding: large-error
outcomes keep its later-time mean and global mean solution errors high. The
median ordering also does not imply paired superiority over every method:
at `t = 0.99`, PACMANN's median is lower than RAD's (0.0970% versus 0.1101%),
but PACMANN has lower error in only three of ten same-seed comparisons.
This observation does not establish a diffusion benefit.

Each time-slice score covers the whole spatial domain. It cannot establish
that the improvement occurs specifically inside the front, or that solution
sharpness caused it. These are physical-time profiles of final models, not
training trajectories. Each relative error uses the reference norm at that
time, which changes as the solution decays; the derived table also provides
absolute RMSE. Reference accuracy, per-viscosity tuning, and changes in solution
amplitude and evolution limit the interpretation. Spatially localized error
analysis is motivated by this observation but has not been performed here;
see the [research direction](research_context.md#motivation-for-spatially-localized-evaluation).

### Reproduction

Reproduce the four figures and five derived CSV tables from the repository root
with the [documented environment](../README.md#setup-and-execution):

```sh
venv/bin/python -m scripts.analysis.burgers_regime_comparison
```

Figures go to `docs/figures/burgers_regime_comparison/` and are tracked. Regenerable
tables go to the ignored `output/burgers/regime_comparison/`.
`final_metrics.csv` and `final_summary.csv`
contain the individual scores and aggregates; `paired_seed_ratios.csv` and
`paired_summary.csv` retain the same-seed contrasts; `physical_time_errors.csv`
contains per-time relative L2 (%) and absolute RMSE. Solution differences in
the paired summary are percentage points; residual differences retain RMS units.
