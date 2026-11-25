# Burgers adaptive diffusion: ν = 0.01/π, α = 1

The [shared protocol and execution conditions](../../nu_0p0031830989/alpha_0p1/README.md)
define the comparison, archived controls, selection rule, and reproduction command.
This candidate's complete scientific settings are in
[the configuration](../../../../config/burgers/nu_0p0031830989/burgers_nu_0p0031830989_alpha_1.yaml).
See the [cross-viscosity development results](../../../../docs/results.md#adaptive-diffusion-development)
for the selected settings and limitations.

## Final outcomes

All three planned development seeds completed. Solution relative L2 is reported
in percent; PDE-residual RMS is unnormalized. Aggregates retain every seed and
use the arithmetic mean and sample standard deviation.

| Seed | Final solution relative L2 (%) | Final PDE-residual RMS |
| --- | ---: | ---: |
| [3000](pacmann_sharpness_diffusion/seed_3000/metrics.json) | 0.092611 | 0.0042348 |
| [3001](pacmann_sharpness_diffusion/seed_3001/metrics.json) | 0.0702095 | 0.00354918 |
| [3002](pacmann_sharpness_diffusion/seed_3002/metrics.json) | 0.0429846 | 0.00650969 |
| Mean ± sample SD | 0.0686017 ± 0.0248522 | 0.00476455 ± 0.00154972 |

The [study-level comparison](../../../../docs/results.md#adaptive-diffusion-development)
selects α = 1 at this viscosity using all four finite candidates and the
[archived constant-diffusion control](../sigma_1e-4/README.md).
Ordinary PACMANN remains a comparison baseline.
