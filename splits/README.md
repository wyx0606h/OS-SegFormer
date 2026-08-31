# Reproduction manifests

These CSV files contain sample identifiers, split names, and paths relative to
the user-provided dataset root. They do not contain images or annotations.

- `floodnet/sup398.csv`: FloodNet paper protocol, with 398 train, 450
  validation, and 448 test rows.
- `rescuenet/official.csv`: RescueNet official split, with 3595 train, 449
  validation, and 450 test rows.
- `rescuenet/summary.json`: frozen RescueNet manifest count and row-hash
  summary.

Do not use the test split for checkpoint or hyperparameter selection.
