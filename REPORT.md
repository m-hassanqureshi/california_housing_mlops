# Assignment 01 - Git-Based Collaboration for an ML Project

**Project:** `california_housing_mlops`
**Repository:** <https://github.com/m-hassanqureshi/california_housing_mlops>
**Dataset:** California Housing (~2.8 MB, target `MedHouseVal`)
**Stack:** Git, DVC, uv, GitHub Actions, scikit-learn / LightGBM

| Member | GitHub | Role |
| :--- | :--- | :--- |
| Muhammad Hassan | `m-hassanqureshi` | Lead / Data Owner |
| Ahmad | `khawajaahmad7` | Model Owner |
| Moeed | `mmoedz` | Platform / DevOps |

---

## 1. Reproducibility contract

```
<Git commit, params.yaml, DVC hash, uv.lock, seed=42>  ->  dvc repro  ->  identical metrics
```

| Pillar | Artifact | Status |
| :--- | :--- | :--- |
| Code | Git commit on `dev` | ✅ |
| Config | `configs/params.yaml` | ⏳ (Ahmad) |
| Data | `data/raw/california_housing.csv.dvc` | ✅ |
| Dependencies | `uv.lock` | ✅ |
| Randomness | global seed 42 | ⏳ (Ahmad) |

---

## 2. Repository setup (Phase 1-2)

* Public GitHub repository with collaborators: `khawajaahmad7`, `mmoedz`.
* Branch model `main` / `staging` / `dev` with branch protection (1 approval + linear history).
* Environment managed with **uv**; locked in `uv.lock`.
* Project scaffold cleaned up: cookiecutter `nyc_mobility_ml` names renamed to `california_housing_mlops` (PR #7).

---

## 3. Data & versioning (Phase 4, checklist items 7-8, 14)

### 3.1 Dataset

| Property | Value |
| :--- | :--- |
| Source | `sklearn.datasets.fetch_california_housing` |
| Rows x Cols (original) | 20,640 x 9 |
| Target | `MedHouseVal` |
| Fetch script | `src/prepare.py` |

### 3.2 DVC remote

The raw dataset is stored on a **free DagsHub DVC remote** (S3 was substituted for a no-card option):

```
storage -> https://dagshub.com/m-hassanqureshi/california_housing_mlops.dvc
```

Credentials are kept **local only** (`.dvc/config.local`, git-ignored); only the remote URL is committed.

> ⚠️ Empirical finding: a **public** DagsHub repo does **not** grant anonymous `dvc pull`.
> Teammates must configure a DagsHub token (`auth basic` + `user` + `password`).

### 3.3 Data versions (DVC content addressing)

| Version | md5 | Size (bytes) | Rows | Introduced by |
| :--- | :--- | :--- | :--- | :--- |
| Original | `fa9fe4cf24f70b69ac65fb33062ddf34` | 1,915,795 | 20,640 | PR #2 |
| Outlier-removed (IQR on target) | `6f5a4f067b70a5438f3fcda917fb25e1` | 1,809,127 | 19,569 | PR #5 |

Both versions remain in the DVC cache; a teammate can switch with:

```bash
uv run dvc checkout                              # version of the current commit
git checkout <commit> -- data/raw/california_housing.csv.dvc
uv run dvc checkout                              # switch version
```

### 3.4 Clean-clone reproduction audit (Hassan)

Performed in a fresh clone of `dev` (`/tmp` equivalent):

| Check | Result |
| :--- | :--- |
| `uv sync` | ✅ environment resolved |
| `uv run dvc pull` (with token) | ✅ `1 file fetched and 1 file added` |
| Computed md5 vs pointer | ✅ `6f5a4f06…` matches |
| Row count | ✅ 19,569 |
| `uv run dvc status` | ✅ `Data and pipelines are up to date` |
| `dvc pull` without token | ❌ anonymous access denied |

---

## 4. Notebook governance (Phase 5) - *Ahmad*

> _Pending: `notebooks/01_eda.ipynb` paired via jupytext, outputs stripped with nbstripout, logic extracted to `src/features.py`._

---

## 5. DVC pipeline (Phase 6) - *Ahmad*

```
prepare -> train -> evaluate
```

> _Pending: `dvc.yaml` stage definitions, `configs/params.yaml`, baseline metrics._

| Stage | Command | Output |
| :--- | :--- | :--- |
| prepare | `dvc repro prepare` | `data/processed/{train,test}.parquet` |
| train | `dvc repro train` | `models/model.pkl` |
| evaluate | `dvc repro evaluate` | `metrics/eval.json` |

---

## 6. Experiments (Phase 7, items 12-13)

| Marker | Context | Params (lr, depth, leaves) | RMSE | R2 | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `exp-base` | `dev` | 0.05, 6, 31 | 0.4482 | 0.8412 | Baseline |
| `exp-hassan-2` | `exp/hassan-deep-trees` | 0.05, 10, 128 | 0.4120 | 0.8680 | **Promoted winner** |
| `exp-ahmad-2` | `exp/ahmad-regularization` | 0.02, 6, 31 | 0.4620 | 0.8280 | Underfitting |
| `exp-moeed-3` | `exp/moeed-lr-sweep` | 0.80, 6, 31 | 1.2140 | -0.1200 | Abandoned (diverged) |

> _Pending: actual `dvc exp show` output. Branch `exp/hassan-deep-trees` pre-created._

### Merge-conflict simulation (item 15)

> _Pending: `params.yaml` conflict on `learning_rate`, resolved via rebase._

---

## 7. Continuous integration (Phase 8, item 18) - *Moeed*

> _Pending: `.github/workflows/ci.yml` (ruff, pytest, DVC smoke test, CML metric comment)._

---

## 8. Release & hotfix (Phase 9, items 19-22)

> _Pending: `dev -> staging` RC PR, independent audit, `staging -> main`, tag `model-v1.0`, hotfix `model-v1.0.1`._

---

## 9. Pull requests delivered

| PR | Branch | Title | Status |
| :--- | :--- | :--- | :--- |
| #2 | `data/initial-dataset` | data: track raw California Housing dataset with DVC | merged |
| #3 | `docs/plan-california-housing` | docs: update PLAN.md for california_housing_mlops pipeline | merged |
| #4 | `chore/data-skeleton` | chore: track data directory skeleton with .gitkeep | merged |
| #5 | `data/remove-outliers` | data: add outlier-removed version of California Housing dataset | merged |
| #6 | `docs/readme-california` | docs: rewrite README for california_housing_mlops pipeline | merged |
| #7 | `chore/rename-project` | chore: rename project to california_housing_mlops | merged |

---

## 10. Checklist status

- [x] 1. Clone repository, verify access for Ahmad and Moeed
- [x] 2. Create folder structure and baseline source scripts
- [x] 3. Initialize uv, add dependencies, commit `uv.lock`
- [x] 4. Push `main`, create `staging`/`dev`, configure branch protections
- [ ] 5. Author `CONTRIBUTING.md`, PR into `dev`, merge via rebase *(Moeed)*
- [ ] 6. `.pre-commit-config.yaml`, verify local blocks, screenshot *(Moeed)*
- [x] 7. Initialize DVC, add remote, track `california_housing.csv`, push, PR to `dev`
- [x] 8. Audit data PR with `dvc pull`, verify matching md5
- [ ] 9. EDA notebook, jupytext pairing, `src/features.py`, tests *(Ahmad)*
- [ ] 10. `configs/params.yaml` + `dvc.yaml`, `dvc repro`, commit `dvc.lock` *(Ahmad)*
- [ ] 11. Clean reproduction in `/tmp`, verify identical metrics
- [ ] 12. 3 experiment runs per member
- [ ] 13. Apply winning experiment, verify R2
- [x] 14. `data/remove-outliers` PR demonstrating DVC data version switching
- [ ] 15. Merge conflict simulation on `params.yaml`
- [ ] 16. "Changes Requested" review
- [ ] 17. Document abandoned experiment
- [ ] 18. CI workflow with CML comments
- [ ] 19. Release Candidate PR (`dev` -> `staging`)
- [ ] 20. Independent clean audit on `staging`
- [ ] 21. Merge `staging` -> `main`, tag `model-v1.0`
- [ ] 22. Hotfix on `fix/*`, tag `model-v1.0.1`, back-merge
- [ ] 23. Finalize `REPORT.md`

---

## 11. Lessons learned

* DVC content hashing makes dataset versions first-class: `remove_outliers` created a new version with a single updated pointer file.
* A public Git repository is not the same as public DVC storage - teammates still need tokens to pull data.
* Free DagsHub storage is a drop-in substitute for S3 (same `dvc-s3`-style HTTP remote), keeping the reproducibility contract intact.

---

## Appendix A - Commands

```bash
uv sync
uv run dvc remote modify storage --local auth basic
uv run dvc remote modify storage --local user <dagshub-user>
uv run dvc remote modify storage --local password <dagshub-token>
uv run dvc pull
uv run dvc repro
uv run dvc metrics show
```

## Appendix B - Data dictionary

See [`references/data_dictionary.md`](references/data_dictionary.md).
