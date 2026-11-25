# Burgers adaptive diffusion: ν = 0.001, α = 0.1

The [shared protocol and execution conditions](../../nu_0p0031830989/alpha_0p1/README.md)
define the comparison, archived controls, selection rule, and reproduction command.
This candidate's complete scientific settings are in
[the configuration](../../../../config/burgers/nu_0p001/burgers_nu_0p001_alpha_0p1.yaml).
See the [cross-viscosity development results](../../../../docs/results.md#adaptive-diffusion-development)
for the selected settings and limitations.

## Final outcomes

All three planned development seeds completed. Solution relative L2 is reported
in percent; PDE-residual RMS is unnormalized. Aggregates retain every seed and
use the arithmetic mean and sample standard deviation.

| Seed | Final solution relative L2 (%) | Final PDE-residual RMS |
| --- | ---: | ---: |
| [3000](pacmann_sharpness_diffusion/seed_3000/metrics.json) | 57.2536 | 7.20878 |
| [3001](pacmann_sharpness_diffusion/seed_3001/metrics.json) | 61.5273 | 7.88505 |
| [3002](pacmann_sharpness_diffusion/seed_3002/metrics.json) | 33.8041 | 2.78278 |
| Mean ± sample SD | 50.8617 ± 14.9261 | 5.95887 ± 2.77128 |

The [study-level comparison](../../../../docs/results.md#adaptive-diffusion-development)
selects α = 10 at this viscosity using all four finite candidates and the
[archived constant-diffusion control](../sigma_1e-6/README.md).
Ordinary PACMANN remains a comparison baseline.
