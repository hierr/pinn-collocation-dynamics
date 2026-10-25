# Research context

The study asks whether residual-gradient adaptive collocation benefits more from
sharp solution features, and whether stochastic diffusion improves PACMANN.
Neither benefit is established. Solution sharpness and coordinate gradients of
the squared residual are different quantities.

This persistent branch supports the study by validating reference data. The
`hyperparameter-tuning` branch retains scientific development; `main` retains
accepted implementations, conclusions, and full frozen-setting evaluations.
Transfer selected changes and cite the supporting commit rather than merging
entire workstreams and their artifacts.

## Burgers reference validation

Before sweeping viscosity, compare the collaborator's Chebyshev-collocation/Radau
solutions at `0.01/pi` with the original PACMANN reference. Use the original
256 spatial points and 100 times through `0.99`, preserving the original values.
Interpolate generated solutions in space only, at exact shared snapshot times.

Check both reference disagreement and the effect on measured PINN error using
three existing fixed PACMANN models. No retraining is needed because these
reference datasets are used only for evaluation. The checks support N1024 at
this viscosity and time interval; see [results](results.md).

## Subsequent study

Generate and validate references over a viscosity range representing smooth,
intermediate, and sharp solutions. Tune diffusion separately at each viscosity
using development seeds; freeze settings before ten-seed comparisons of fixed
Hammersley, upstream random resampling, RAD, PACMANN, and PACMANN with diffusion.
This estimates performance after local tuning; it does not test transfer of one
fixed sigma across viscosities. Other PDE problems can follow later.

Viscosities, sharpness measures, tuning grids, budgets, matched point populations,
and evaluation seeds require a concrete approved protocol before execution.
Preserve failures and report variability. New viscosities require numerical
accuracy checks; this validation does not certify the whole parameter range.

The sharper `nu = 0.001` extension remains at reference validation. Spatial
refinement from N1024 to N2048 materially changes the steep front and its
gradients; see the [resolution check](../output/burgers_reference_validation/nu_0p001_n1024_n2048/README.md).
Establish adequate reference resolution before tuning. The density of the PINN
evaluation grid is a separate check that remains necessary for this sharper case.
