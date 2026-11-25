# Burgers adaptive diffusion: ν = 0.1, α = 100

The [shared protocol and execution conditions](../../nu_0p0031830989/alpha_0p1/README.md)
define the comparison, archived controls, selection rule, and reproduction command.
This candidate's complete scientific settings are in
[the configuration](../../../../config/burgers/nu_0p1/burgers_nu_0p1_alpha_100.yaml).
See the [cross-viscosity development results](../../../../docs/results.md#adaptive-diffusion-development)
for the selected settings and limitations.

## Final outcomes

All three planned development seeds completed. Solution relative L2 is reported
in percent; PDE-residual RMS is unnormalized. Aggregates retain every seed and
use the arithmetic mean and sample standard deviation.

| Seed | Final solution relative L2 (%) | Final PDE-residual RMS |
| --- | ---: | ---: |
| [3000](pacmann_sharpness_diffusion/seed_3000/metrics.json) | 0.0110079 | 0.000810664 |
| [3001](pacmann_sharpness_diffusion/seed_3001/metrics.json) | 0.0123944 | 0.00104744 |
| [3002](pacmann_sharpness_diffusion/seed_3002/metrics.json) | 0.0171968 | 0.0014118 |
| Mean ± sample SD | 0.013533 ± 0.00324775 | 0.00108997 ± 0.000302817 |

The [study-level comparison](../../../../docs/results.md#adaptive-diffusion-development)
selects constant at this viscosity using all four finite candidates and the
[archived constant-diffusion control](../sigma_1e-5/README.md).
Ordinary PACMANN remains a comparison baseline.
