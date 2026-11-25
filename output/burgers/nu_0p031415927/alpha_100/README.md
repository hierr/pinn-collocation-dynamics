# Burgers adaptive diffusion: ν = π/100, α = 100

The [shared protocol and execution conditions](../../nu_0p0031830989/alpha_0p1/README.md)
define the comparison, archived controls, selection rule, and reproduction command.
This candidate's complete scientific settings are in
[the configuration](../../../../config/burgers/nu_0p031415927/burgers_nu_0p031415927_alpha_100.yaml).
See the [cross-viscosity development results](../../../../docs/results.md#adaptive-diffusion-development)
for the selected settings and limitations.

## Final outcomes

All three planned development seeds completed. Solution relative L2 is reported
in percent; PDE-residual RMS is unnormalized. Aggregates retain every seed and
use the arithmetic mean and sample standard deviation.

| Seed | Final solution relative L2 (%) | Final PDE-residual RMS |
| --- | ---: | ---: |
| [3000](pacmann_sharpness_diffusion/seed_3000/metrics.json) | 0.0101436 | 0.000788646 |
| [3001](pacmann_sharpness_diffusion/seed_3001/metrics.json) | 0.0243476 | 0.00127046 |
| [3002](pacmann_sharpness_diffusion/seed_3002/metrics.json) | 0.0174166 | 0.00129228 |
| Mean ± sample SD | 0.0173026 ± 0.0071027 | 0.00111713 ± 0.000284683 |

The [study-level comparison](../../../../docs/results.md#adaptive-diffusion-development)
selects α = 10 at this viscosity using all four finite candidates and the
[archived constant-diffusion control](../sigma_1e-4/README.md).
Ordinary PACMANN remains a comparison baseline.
