# Experimental results

This document consolidates accepted comparisons and brief interpretation.
Detailed validation and tuning artifacts remain on their respective branches;
full frozen-setting evaluation campaigns will be retained on `main`.

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
settings did not imply equal realized computation. Evaluate the selected
candidate on new seeds.

Detailed comparisons and artifacts remain on `hyperparameter-tuning`, source
commit [`7fffcb6`](https://github.com/hierr/pinn-collocation-dynamics/commit/7fffcb600c62208958d5681d1bcf3cf5a8618148).

## Evaluation status

No full frozen-setting ten-seed evaluation has been completed. The N1024
selection applies only at `nu = 0.01/pi`; other viscosities require their own
reference checks and development tuning. See the [study plan](research_context.md).
