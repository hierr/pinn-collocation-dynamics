# Research methodology

This document defines reusable guidelines for designing and evaluating the study's
experiments. The study plan and accepted scientific choices belong in
[research context](research_context.md); campaign-specific protocols and
interpretation belong in [experimental notes](experimental_notes.md); concise
outcome comparisons belong in [results](results.md). The
[output dictionary](output_format.md) explains generated artifacts.

## Work classification and campaign records

Classify work by its intended question before observing its outcome:

- Engineering validation checks whether implementation, dependencies, execution,
  or saved artifacts work. Smoke tests and readiness-only pilots belong here,
  regardless of their duration or training budget. Keep final environment
  choices here; omit validation history from scientific notes and results.
- Scientific development selects or investigates a method, parameter, or design
  choice. Tuning is scientifically relevant: retain its procedure, outcomes,
  selection rationale, and limitations separately from final evaluation.
- Scientific evaluation tests a research question with frozen settings and a
  predeclared comparison. It provides the main evidence for reported claims.

For scientific campaigns, prepare a proportional protocol in experimental notes
with the objective or hypothesis, controls, changed variables, seeds, budgets,
evaluation metrics, selection or acceptance criteria, and decisions supported
by possible outcomes. Obtain approval before execution. Group runs that answer
the same objective under that protocol; do not write a separate prose entry for
every seed. After execution, add results, interpretation, limitations, material
deviations, decisions, and next steps. Record tuning outcomes even when the
selected method does not improve performance. Do not reclassify an unsuccessful
scientific experiment as an engineering check after seeing its results.

Keep generated artifacts as the detailed numerical record. Maintain concise
comparison tables in results with links to notes and artifacts; distinguish
development comparisons from final evaluation. Update research context when
the study plan or a durable scientific decision changes. Create report or paper
drafts only when that deliverable is requested.

## Experiment design

State the research question and changed variables before running an experiment.
Distinguish reproduction of released code from a modified benchmark study, and
record upstream source pins and deviations. Investigate adaptive sampling and
diffusion benefits as open questions rather than assumed outcomes. Validate the
implementation and reference solution before a scientific campaign; reference
solutions must match the PDE parameters being studied.

For generated numerical reference solutions, document the solver, governing
parameters, discretization, tolerances, and evidence of sufficient numerical
accuracy. Retain the generation procedure and identify the resulting dataset.
Distinguish reference-solution error from PINN approximation error.

## Controlled comparisons

Pair methods using the same planned seeds, network initialization, and initial
point design where applicable. Keep problem definition, network, precision,
constraints, and training settings fixed unless they are variables under study.
In matched sampling comparisons, move only physics points and keep boundary,
initial, and supervised constraints fixed. Match point and training budgets;
report actual optimizer work and disclose intentional reproduction deviations.

## Hyperparameter and diffusion tuning

Separate exploratory tuning from final evaluation. Document the search space,
tuning budget, selection criterion, and any changes to the tuning procedure.
Freeze selected settings before evaluating on held-out seeds and, where relevant
to the claim, held-out problems or parameter values. Report tuning cost separately
from evaluation cost.

For a single-benchmark study, evaluate candidate sigmas with matched PACMANN
controls and common tuning seeds, including sigma zero. Select by the predeclared
metric and tie rule, then freeze the setting for separate evaluation seeds.
This establishes a benchmark-specific choice; it does not require reference
data for other parameters and does not establish transfer to them.

When the objective is performance across a parameter range with fixed method
settings, use the following procedure:

1. Define the problem family and intended parameter range. Before searching,
   choose a small development set spanning relatively smooth, intermediate, and
   sharp solutions, using reference solutions of sufficient numerical accuracy.
2. Evaluate the same candidate sigmas on every development condition with the
   same tuning seeds. Include ordinary PACMANN as the sigma-zero baseline. Keep
   coordinate-Adam settings, network, training, constraints, and point budgets
   matched between PACMANN and PACMANN with diffusion for each condition.
3. Select sigma using a predeclared aggregate score that gives each development
   condition equal weight. Normalize errors against the matched PACMANN baseline
   within each condition so raw error magnitudes do not let the hardest condition
   dominate selection. Specify the exact aggregation and tie rule in the campaign
   protocol; sigma zero remains a valid selection.
4. Select one sigma per PDE family and freeze it and the other selected method
   settings before final repeated-seed evaluation. Use evaluation seeds distinct
   from tuning seeds and disclose any prior inspection of evaluation runs. Keep
   each method's settings fixed across
   the parameter sweep so changes in performance can be studied as the problem
   becomes sharper.
5. Include parameter values excluded from tuning when studying transfer across
   the parameter range. New seeds on development conditions assess repeatability
   on those conditions; they do not establish transfer to unseen parameters or
   problem families.

For the diffusion comparison, compare PACMANN with and without the selected noise
while retaining matched coordinate-Adam settings. For the sharpness comparison,
compare adaptive methods against the sampling baselines across the parameter
range. Any additional hyperparameter tuning must follow an explicit, fair policy
for the competing methods, with search budgets and costs disclosed. Preserve
upstream settings when the objective is released-code reproduction.

Different PDE families may use separately selected coefficients. Per-condition
sigma tuning is an optional follow-up answering how well diffusion
can perform after tuning for each particular condition. Keep this study distinct
from the main fixed-setting comparison and use separate tuning and evaluation
seeds. Inconsistent or disappointing results with a shared sigma do not establish
that diffusion cannot help under other settings. A single-benchmark sigma search
provides a benchmark-specific candidate and an initial search range; broader
selection uses a representative development set.

## Evaluation and interpretation

Evaluate solution error on fixed reference or probe sets independent of training
collocation points. Declare metrics and checkpoint-selection rules before runs;
distinguish final errors from training-loss-selected errors and minimum observed
test errors. Define feature regions independently of adaptive point placement
when studying sharp features, and report both global and feature-region errors.

Report the planned and completed run counts, individual seed outcomes, and
aggregate variability. Identify whether uncertainty summaries describe standard
deviation, standard error, or confidence intervals, and explain their calculation.
Report scientific failures, actual computational effort, and timing definitions.
Retain numerical failures rather than silently replacing seeds.
Aggregate only comparable completed evaluations and disclose missing runs.
An individual seed does not reproduce population statistics or establish method
superiority. Its scientific role follows the purpose declared before execution,
not its seed count alone. Record limitations and decision criteria before
expanding a campaign or generalizing its conclusions.

## Execution, interruptions, and recovery

Execute the approved campaign without repeated permission requests for routine
steps within its scope. Seek approval for material changes to the scientific
protocol, such as new parameters, budgets, selection rules, or replacement
seeds. Monitor at intervals proportional to expected runtime. A change of
operator or monitoring agent should not interrupt a healthy training process.

Distinguish numerical or method failures from external operational interruptions
such as shutdowns or an operator stopping a process. Machine execution status
alone does not establish the cause. Include numerical failures in the scientific
record and apply the predeclared failure policy. Do not count an externally
interrupted attempt as evidence of algorithm failure or add an incident diary
to experimental notes. Report a material effect on comparability or missing
evaluation data when one remains.

Restarting the same planned seed from the beginning with unchanged scientific
settings is operational recovery, not a new experiment. It must follow the
approved recovery policy and preserve completed or scientifically meaningful
artifacts. Saved weights do not support exact training resume. Never overwrite
an existing destination; use a separately identified attempt or authorized
cleanup of disposable incomplete artifacts. The current runners reject occupied
destinations and do not automatically retry, so resolve that practical collision
before relaunching. Regenerate missing analysis from completed training artifacts
without retraining when possible. Do not select attempts by favorable error.

## Reproducibility

Save each run's resolved configuration, seed, code revision, environment versions,
model weights, selected point snapshots, metrics, predictions, history, log, and
timing in a fresh directory. Never overwrite a prior run or alter original
reference data. Use saved configurations and fixed evaluation seeds for
post-processing. Record results, interpretation, and next steps in experiment
notes, with compact comparable tables in results.

Provide documented commands or a small executable driver covering campaign
training and evaluation. Instructions should identify the configuration, code
revision, environment, data preparation, expected outputs, and approximate
computational cost. Support regeneration of reported metrics and figures from
saved artifacts independently of retraining. Run scientific campaigns from a
committed source and configuration revision.

## Reproduction environment

Use Python 3.12 with uv and the local `venv`. The authoritative dependency pins
are `pyproject.toml` and `uv.lock`: DeepXDE 1.14.0, PyTorch 2.7.1 with CUDA 12.8
wheels, NumPy 2.3.1, SciPy 1.16.0, Matplotlib 3.10.3, and PyYAML 6.0.2.
Retain this released-code stack for the initial upstream reproduction. Conda is
unnecessary for the current wheel-based environment. Use the PyTorch backend
and float64. The selected host is an NVIDIA GeForce RTX 5050 Laptop GPU with
driver 595.99.02; archive the actual device and installed versions with each
scientific run.

Environment validation is a setup activity. Record final environment decisions
here; reserve experiment notes and results for scientific experiment runs.

## Run retention and retrieval

Track scientific run directories directly in Git, including numerical failures.
Externally interrupted attempts are operational artifacts; their retention or
cleanup follows the approved recovery policy rather than creating a scientific
failure result. Retain configurations, metadata, logs, model weights, point
snapshots, predictions, histories, and diagnostics under the existing output hierarchy.
Retrieve these files from the repository at the corresponding commit; never
rerun into an existing directory. Reference datasets remain ignored and are
restored using the pinned provenance instructions. Compressed archives and
archive checksum sidecars are not required. Setup-validation and compatibility
outputs are disposable and excluded from scientific records and Git.

## Supporting guidance

These references inform this repository's procedures; they do not prescribe its
file layout or tools:

- [NeurIPS paper checklist](https://neurips.cc/public/guides/PaperChecklist):
  guidance on experimental settings, reproducibility, uncertainty, and compute.
- [Research-code release guidance](https://github.com/paperswithcode/releasing-research-code):
  recommendations for dependencies, training and evaluation code, saved models,
  and documented reproduction commands.
- [Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html):
  selection bias and the distinction between model selection and evaluation.
- [Sivaprasad et al., 2020](https://proceedings.mlr.press/v119/sivaprasad20a.html):
  accounting for hyperparameter tuning effort in method comparisons.
