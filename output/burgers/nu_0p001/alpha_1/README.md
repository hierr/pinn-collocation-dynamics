# Burgers adaptive diffusion: ν = 0.001, α = 1

The [shared protocol and execution conditions](../../nu_0p0031830989/alpha_0p1/README.md)
define the comparison, archived controls, selection rule, and reproduction command.
This candidate's complete scientific settings are in
[the configuration](../../../../config/burgers/nu_0p001/burgers_nu_0p001_alpha_1.yaml).
See the [cross-viscosity development results](../../../../docs/results.md#adaptive-diffusion-development)
for the selected settings and limitations.

## Final outcomes

All three planned development seeds completed. Solution relative L2 is reported
in percent; PDE-residual RMS is unnormalized. Aggregates retain every seed and
use the arithmetic mean and sample standard deviation.

| Seed | Final solution relative L2 (%) | Final PDE-residual RMS |
| --- | ---: | ---: |
| [3000](pacmann_sharpness_diffusion/seed_3000/metrics.json) | 44.6065 | 2.42783 |
| [3001](pacmann_sharpness_diffusion/seed_3001/metrics.json) | 18.3838 | 2.33264 |
| [3002](pacmann_sharpness_diffusion/seed_3002/metrics.json) | 68.7051 | 5.67657 |
| Mean ± sample SD | 43.8985 ± 25.1681 | 3.47902 ± 1.90374 |

The [study-level comparison](../../../../docs/results.md#adaptive-diffusion-development)
selects α = 10 at this viscosity using all four finite candidates and the
[archived constant-diffusion control](../sigma_1e-6/README.md).
Ordinary PACMANN remains a comparison baseline.
