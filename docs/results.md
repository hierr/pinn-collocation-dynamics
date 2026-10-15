# Experimental results

This document consolidates scientific comparisons for quick access. Link each
table or figure to its campaign README, retained artifacts, and relevant study phase.
Identify the metric, units, planned/completed seed counts, variability, and
whether a comparison concerns tuning or frozen-setting evaluation. Include brief
interpretation and limitations needed to understand each comparison; broader
study decisions belong in [research context](research_context.md).

## Current evidence

The [original-benchmark sigma comparison](#original-benchmark-burgers-sigma-tuning)
selected `1e-4` under its original training and evaluation protocol. This is
historical development evidence, not a frozen setting for the new controlled
study. No full ten-seed controlled evaluation has been completed. Parameterized
sharpness comparisons require validated references and the
[accepted controls and evaluation policy](research_context.md#accepted-controls-and-evaluation-policy).

<a id="original-benchmark-burgers-sigma-tuning"></a>

## Original-benchmark Burgers sigma tuning

Scientific development: three matched seeds per setting. Final relative L2 is
reported as a percentage; paired change is candidate minus PACMANN in percentage
points, so negative values favor diffusion. Variability is sample SD. All 15
runs completed. Each sigma below links to its run directories and archived
configurations; the [shared design](../output/burgers_sigma_tuning_01_baseline/README.md#comparison-design)
explains the comparison and selection rule.

| Sigma | Seed 2000 (%) | Seed 2001 (%) | Seed 2002 (%) | Mean (%) | Sample SD (%) | Mean paired change (pp) | Improved pairs | Total training seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [0](../output/burgers_sigma_tuning_01_baseline/README.md) | 0.0961529 | 0.0443555 | 0.0930452 | 0.0778512 | 0.0290497 | 0 | 0/3 | 2690.1 |
| [1e-5](../output/burgers_sigma_tuning_01_1e-5/README.md) | 0.237395 | 0.0284003 | 0.0599559 | 0.108584 | 0.112664 | +0.0307324 | 2/3 | 2694.1 |
| [1e-4](../output/burgers_sigma_tuning_01_1e-4/README.md) | 0.0558527 | 0.0450372 | 0.0404108 | **0.0471002** | 0.00792501 | **-0.0307510** | 2/3 | 2672.6 |
| [1e-3](../output/burgers_sigma_tuning_01_1e-3/README.md) | 0.177654 | 28.1237 | 0.0784063 | 9.45993 | 16.1634 | +9.38208 | 1/3 | 2666.8 |
| [1e-2](../output/burgers_sigma_tuning_01_1e-2/README.md) | 0.252734 | 0.0956092 | 0.0920033 | 0.146782 | 0.0917747 | +0.0689310 | 1/3 | 2633.5 |

The predeclared minimum-mean rule selects sigma `1e-4`. Its mean final relative
L2 is lower than PACMANN's, but only two of three matched seeds improve. Treat
this as a benchmark-specific tuning choice for the separate evaluation, not as
evidence of a general diffusion benefit or an effect across solution sharpness.
The high `1e-3`, seed 2001 error is a completed result and is retained.
