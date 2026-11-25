# Burgers adaptive diffusion development

This scientific-development comparison tunes alpha separately at five viscosities,
with the base diffusion coefficient frozen from the preceding sigma sweeps.
Finite candidates are alpha = 0.1, 1, 10, 100 on paired seeds 3000–3002.
Archived ordinary PACMANN and selected constant-diffusion controls are reused.
Constant diffusion is eligible as alpha → infinity. Select by arithmetic mean
final solution relative L2 over all three seeds; exact ties favor constant
diffusion, then larger alpha. Assess residual RMS at the same selected setting.
Retain failures and large errors; incomplete comparisons do not select a winner.
No search expansion is automatic.

All 60 planned adaptive diffusion runs completed without reported execution
failures. They record source revision `4b64662`; historical control revisions
remain in their metadata. The saved method identifier
`pacmann_sharpness_diffusion` denotes PACMANN with adaptive diffusion.

See the [study rationale](../../../../docs/research_context.md#adaptive-diffusion-development),
[artifact dictionary](../../../../docs/output_format.md), and
[study results](../../../../docs/results.md). Candidate-specific settings are in
[this configuration](../../../../config/burgers/nu_0p0031830989/burgers_nu_0p0031830989_alpha_0p1.yaml).

## Execution conditions

Adaptive diffusion candidates use an NVIDIA GeForce RTX 5090 (32 GB),
Python 3.12.16, driver 595.99.02, and the locked DeepXDE 1.14.0 /
PyTorch 2.7.1+cu128 stack.
Five viscosity sweeps share one GPU: three concurrent seeds per sweep, up to
15 training workers total. Each sweep advances through alpha sequentially.
Each worker uses one PyTorch intra-op/inter-op and BLAS/OpenMP thread.
The host has an AMD Ryzen 9 9900X (12 cores / 24 threads) and 121 GiB usable RAM.

The host is shared and sweeps advance independently, so resource contention and
concurrency can vary. Earlier controls used an RTX 5070. Wall times describe
these execution conditions; they do not isolate adaptation overhead or support
direct speed comparisons against the historical controls.

## Reproduction

From the repository root, use the documented pinned environment. Select the
allocated GPU below. Each process overrides only the runner's execution selectors;
scientific settings remain in the complete YAMLs and are archived per run.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PINN_GPU=0
for tag in 0p1 0p031415927 0p01 0p0031830989 0p001; do
  CUDA_VISIBLE_DEVICES="$PINN_GPU" venv/bin/python -u -c \
    "from scripts.run_experiments import run_burgers_alpha_tuning as t; t.NU_TAG='$tag'; t.ACTION='run'; t.main()" &
done
wait
```

Retained runs cannot be overwritten. For a new execution, assign unused campaign
names in the four alpha YAMLs for each viscosity; keep the archived-control
campaign names unchanged. Training and post-processing use the existing campaign
runner. To report the retained runs, keep their original campaign names and
set `NU_TAG` for one viscosity and
`ACTION = "summarize"` in the tuning runner to print all paired outcomes and
the selection; this reporting action never retrains or rewrites scientific artifacts.

## Final outcomes

All three planned development seeds completed. Solution relative L2 is reported
in percent; PDE-residual RMS is unnormalized. Aggregates retain every seed and
use the arithmetic mean and sample standard deviation.

| Seed | Final solution relative L2 (%) | Final PDE-residual RMS |
| --- | ---: | ---: |
| [3000](pacmann_sharpness_diffusion/seed_3000/metrics.json) | 0.0726902 | 0.00621599 |
| [3001](pacmann_sharpness_diffusion/seed_3001/metrics.json) | 0.0743013 | 0.00376489 |
| [3002](pacmann_sharpness_diffusion/seed_3002/metrics.json) | 0.0907508 | 0.00365801 |
| Mean ± sample SD | 0.0792475 ± 0.0099947 | 0.0045463 ± 0.00144698 |

The [study-level comparison](../../../../docs/results.md#adaptive-diffusion-development)
selects α = 1 at this viscosity using all four finite candidates and the
[archived constant-diffusion control](../sigma_1e-4/README.md).
Ordinary PACMANN remains a comparison baseline.
