# Burgers adaptive diffusion: ν = 0.1, α = 0.1

The [shared protocol and execution conditions](../../nu_0p0031830989/alpha_0p1/README.md)
define the comparison, archived controls, selection rule, and reproduction command.
This candidate's complete scientific settings are in
[the configuration](../../../../config/burgers/nu_0p1/burgers_nu_0p1_alpha_0p1.yaml).
See the [cross-viscosity development results](../../../../docs/results.md#adaptive-diffusion-development)
for the selected settings and limitations.

## Final outcomes

All three planned development seeds completed. Solution relative L2 is reported
in percent; PDE-residual RMS is unnormalized. Aggregates retain every seed and
use the arithmetic mean and sample standard deviation.

| Seed | Final solution relative L2 (%) | Final PDE-residual RMS |
| --- | ---: | ---: |
| [3000](pacmann_sharpness_diffusion/seed_3000/metrics.json) | 0.00984315 | 0.000680491 |
| [3001](pacmann_sharpness_diffusion/seed_3001/metrics.json) | 0.0126879 | 0.000797816 |
| [3002](pacmann_sharpness_diffusion/seed_3002/metrics.json) | 0.0282097 | 0.00150087 |
| Mean ± sample SD | 0.0169136 ± 0.0098856 | 0.00099306 ± 0.000443675 |

The [study-level comparison](../../../../docs/results.md#adaptive-diffusion-development)
selects constant at this viscosity using all four finite candidates and the
[archived constant-diffusion control](../sigma_1e-5/README.md).
Ordinary PACMANN remains a comparison baseline.
