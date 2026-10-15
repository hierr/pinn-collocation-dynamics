# Burgers reference check

Check reference suitability at viscosity `0.01/pi` over `[-1, 1] × [0, 0.99]`,
using the original PACMANN dataset, the generated N512/N1024 solutions, and
[three existing PACMANN models](../../../../output/burgers_sigma_tuning_01_baseline/burgers/pacmann_adam/).
Models use their archived final weights and configurations; no training occurs.

## Comparison

Use the original 100 snapshot times, present exactly in both generated datasets.
Compare reference solutions on the original spatial grid using Chebyshev
barycentric interpolation. Evaluate each saved model directly on all three
native spatial grids and against all references on the original grid.
Verify restored original-grid predictions against the archived predictions.
Evaluation uses CPU float64; the models were trained on the RTX 5050.

- [Reference errors](reference_errors.csv): relative L2, MSE, and maximum absolute
  difference on the original grid. The named reference supplies the denominator.
- [Model errors](model_errors.csv): the same measures per seed and dataset;
  `grid` distinguishes original-grid comparisons from native-grid evaluation.
- [Diagnostic figure](comparison.png): per-time reference differences and
  original-grid model errors.

CSV relative L2 values are fractions; multiply by 100 for percentages. Native
Chebyshev and original uniform-grid scores have different spatial weighting.

## Finding

N1024 differs from PACMANN by 0.000319% relative L2 on the common grid and changes
the three model error scores by less than 0.006% relative to their original
scores. N512 differs by 0.0664% and increases those scores by approximately
22–80%. These results support using N1024 for this case; N512 is not an equally
accurate replacement at the error levels reached by these models.

This is a consistency check against an existing reference, not an independent
accuracy certificate or evidence for other viscosities. Sharper generated cases
need their own resolution and time-integration accuracy checks. Retain these
detailed artifacts on the evaluation branch and the main conclusion in the
main branch's research documents.

## Execution

The [standalone script](../../../../scripts/post_processing/validate_burgers_reference.py)
contains the fixed input paths and seeds. From the repository root:

```bash
MPLBACKEND=Agg venv/bin/python -m scripts.post_processing.validate_burgers_reference
```

It writes into a new directory and refuses to overwrite an existing review;
select an unused `OUTPUT` path at the top of the script to repeat the check.
Input datasets and source runs remain unchanged. This branch is based on
`0c766bc`; source-run metadata records the original training revision and stack.
