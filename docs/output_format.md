# Artifact dictionary

The compact review branch retains original records for all 250 final evaluation
runs. The [results](results.md) synthesize them, and the
[review guide](peer_review.md) explains the experimental design.

## Retained evidence

Each campaign directory contains a concise README with per-seed tables and
method aggregates. Each `<method>/seed_<seed>/` directory retains:

| File | Meaning |
| --- | --- |
| `config.yaml` | Original resolved settings; `selected_method` and `selected_seed` identify this run. The archived `seeds` list contains all planned campaign seeds. |
| `metadata.json` | Training source revision, installed package versions, reference SHA-256, and PyTorch device. |
| `status.json` | Original training execution state. `complete` means training finished; it is not an accuracy or superiority judgment. |
| `metrics.json` | Final-model error scores, training seconds, and sampling-event count. |

Earlier campaigns retain their original paths under `output/burgers/<campaign>/`.
The nu = 0.001 campaign uses `output/burgers/nu_0p001/<campaign>/`.

One complete illustrative run is retained at:

```text
output/burgers/burgers_nu_0p0031830989_10seed/pacmann_adam/seed_4000/
```

It is the first planned evaluation seed at that viscosity and is a successful
example. It does not summarize the method's full outcome distribution. Other
runs' complete artifacts are available at the
[fixed research checkpoint](https://github.com/hierr/pinn-collocation-dynamics/tree/6e7a638383fe0e04d9d999a2cc491c313f587cc8/output).

## Numerical metrics

Let `error = prediction - reference`, with one value at each coordinate.

| Metric | Definition and units |
| --- | --- |
| `final_mse` | `mean(error**2)` on the uniform solution evaluation grid. |
| `final_relative_l2` | `sqrt(sum(error**2) / sum(reference**2))`; a dimensionless fraction. Multiply by 100 for the percentages in the result tables. |
| `final_residual_rms` | RMS of `u_t + u*u_x - nu*u_xx`, differentiated through the trained network on an independent interior midpoint grid. It does not use reference solution values. |
| `boundary_rmse` | RMSE against zero at both spatial boundaries, with equal weights over boundary coordinates and evaluation times. |
| `initial_rmse` | RMSE against `-sin(pi*x)` at the initial time on the uniform spatial grid. |
| `training_seconds` | Compilation and network optimization time, including sampling callbacks and CUDA synchronization; excludes final evaluation and plotting. |
| `movement_events` | Number of sampler updates. One event contains a full population replacement or all configured coordinate-Adam steps. Zero for fixed Hammersley. |

Reference data are used only for evaluation. The five frozen campaigns use
1,024 uniform spatial points and 199 exact saved times through `t=0.99`, with
barycentric interpolation of reference values in space. PDE residuals use a
separate 1,024×200 midpoint grid. The demo uses its own smaller evaluation grids,
recorded in `config/peer_review.yaml` and its resolved run configuration.

## Files in the complete example and generated demo

| File | Meaning |
| --- | --- |
| `models/final.pt` | Network state dictionary for independent evaluation; not an exact training-resume checkpoint. |
| `models/phase_<NN>_<optimizer>.pt` | Network weights at an optimizer-stage endpoint. |
| `predictions.npz` | `coordinates` has shape `(n, 2)` with columns `[x, t]`; `reference` and `prediction` each have shape `(n, 1)`. Coordinates form a complete time-major Cartesian grid. |
| `points/<snapshot>.npz` | `interior` is the adaptive physics population; `pde` is the complete PDE-residual population; `constraints` contains fixed initial/boundary coordinates. All coordinates have columns `[x, t]`. |
| `history.npz` | Recorded global optimizer `steps` and component arrays `loss_train` and `loss_test`. These are optimization losses, not the final independent solution and residual metrics. |
| `training.json` | Per-stage optimizer work and elapsed time. Contains `phase`, `optimizer`, `seconds`, `global_step`, `optimizer_iterations`, `movement_events`, and `pde_points`; L-BFGS stages additionally include `function_evaluations`. |
| `run.log` | Training progress and errors. |
| `diagnostics.json` | Maximum absolute solution error, its coordinate, and per-time relative L2 values. |
| `diagnostics.png` | Training/test losses, final absolute error, final interior points over the reference solution, and reference/prediction profiles. |

Only interior physics points adapt, and only during network Adam. Constraint
coordinates remain fixed; all physics points remain fixed during L-BFGS.
Actual L-BFGS work can exceed nominal limits because of the retained DeepXDE
driver's batching; see [implementation provenance](../references/upstream.md).

The short demo writes new, ignored outputs under a `peer_review_demo*` campaign
and rejects occupied destinations.
