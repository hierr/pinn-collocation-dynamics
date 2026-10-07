# Experimental results

## Burgers reproduction pilot — seed 1000

Full-budget released-code reproduction; five alternating Adam/L-BFGS phases.
Primary L2 is evaluated at minimum recorded training loss. L2 columns are
percentages. These are single-seed outcomes, not estimated means or superiority
claims. See [experiment notes](experimental_notes.md) for interpretation.

| Method | Primary L2 (%) | Final L2 (%) | Final MSE | Minimum observed L2 (%) | Training seconds | Actual L-BFGS iterations | Movement events |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Hammersley | 0.182490 | 0.202476 | 1.547251e-6 | 0.0862134 | 978.1 | 16,163 | 0 |
| PACMANN-Adam | 0.0737324 | 0.0737324 | 2.051779e-7 | 0.0440476 | 906.1 | 15,620 | 700 |

Both methods used 35,000 Adam steps and passed artifact/invariant/reload checks.
Hammersley retains 2,740 PDE coordinates; PACMANN uses 2,500 after its first
update. L-BFGS counts include overshoot. Timings exclude evaluation reload and
post-processing. Published Table 2 ten-run mean L2 errors are 0.61% / 0.07%
respectively; they are descriptive context, not single-seed thresholds.

Local artifacts: `output/burgers_pilot_01`. Expansion to ten seeds awaits review.
