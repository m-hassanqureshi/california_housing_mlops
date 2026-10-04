## What changed and why

<!-- One or two sentences. Link the PLAN.md step or checklist item if there is one. -->

## Metrics (before → after)

<!-- For pipeline, params or data changes: copy from `uv run dvc metrics diff dev`
     or the CML comment. Write "n/a" for docs/tooling-only PRs. -->

| Metric | `dev` | This PR |
| :--- | :---: | :---: |
| RMSE | | |
| MAE | | |
| R² | | |

## ML review checklist

- [ ] **No data leakage:** the target and anything derived from it are excluded from the features.
- [ ] **Preprocessing scope:** scalers, imputers and encoders are fit on the training split only.
- [ ] **Portability:** paths use `pathlib` relative to the repo; no absolute or user-specific paths.
- [ ] **Determinism:** `base.seed` (42) is used for every split and model.
- [ ] **Metrics:** computed on the held-out test split with the shared `src/evaluate.py`.
- [ ] **DVC:** `dvc push` ran before this PR was opened (`dvc status -c` is clean).
- [ ] **Notebooks:** paired with jupytext and stripped of outputs and execution counts.
- [ ] **Checks:** `uv run pre-commit run --all-files` and `uv run pytest` pass locally.
