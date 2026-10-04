# Contributing

How the three of us work in this repository. If something here disagrees with
`PLAN.md`, this file wins; update both in the same PR.

## 1. Branches

Nobody pushes directly to `main`, `staging` or `dev`. Every change reaches them
through a reviewed pull request.

| Branch | Branched from | Merges into | Purpose |
| :--- | :--- | :--- | :--- |
| `main` | — | — | Tagged production releases only (`model-v1.0`, `model-v1.0.1`, …) |
| `staging` | — | `main` | Release candidates, audited on a clean clone before promotion |
| `dev` | — | `staging` | Integration branch; all day-to-day work lands here |
| `feat/<name>` | `dev` | `dev` | Code, config and tooling changes |
| `data/<name>` | `dev` | `dev` | Dataset changes (new `.dvc` pointer + `dvc push`) |
| `exp/<member>-<idea>` | `dev` | **never merged** | Personal experiment sandbox |
| `fix/<name>` | `main` | `main`, then `dev` via PR | Production hotfixes |
| `chore/<name>`, `docs/<name>` | `dev` | `dev` | Maintenance and documentation |

Rules:

* Open feature, data, chore and docs PRs against **`dev`**, never against `main`.
  `main` only receives PRs from `staging` and `fix/*`.
* Keep one topic per branch and delete the branch after it merges.
* A successful experiment is promoted by applying its parameters on a new
  `feat/*` branch from `dev`. The `exp/*` branch itself stays unmerged.
* Commit each experiment run on its `exp/*` branch (`params.yaml`, `dvc.lock`,
  `metrics/eval.json`). `dvc exp run` alone stores hidden refs and the pushed
  branch would look identical to `dev`.
* An abandoned `exp/*` branch gets an `ABANDONED.md` with the measured numbers
  and the reason it was dropped.

## 2. Merging

| PR | Merge button |
| :--- | :--- |
| `feat/*`, `data/*`, `chore/*`, `docs/*` → `dev` | **Rebase and merge** |
| `dev` → `staging`, `staging` → `main`, `fix/*` → `main` | **Create a merge commit** |
| hotfix back to `dev` (`chore/sync-<tag>` with the fix cherry-picked) | **Rebase and merge** |

Promotions use merge commits on purpose. A rebase or squash merge rewrites the
commits, so `staging` and `main` would stop sharing history with `dev` and every
later release PR would conflict on files nobody changed.

Squash merging is only for PRs whose intermediate commits don't build.

### Branch protection (repository admin)

Settings → Branches, for `main`, `staging` and `dev`:

- [x] Require a pull request before merging, with **1 approval**
- [x] Dismiss stale pull request approvals when new commits are pushed
- [x] Require status checks to pass: **`quality-and-smoke-train`** (selectable
  after the CI workflow has run once)
- [x] Do not allow bypassing the above settings
- [x] Block force pushes and deletions

On **`dev` only**, also tick **Require linear history**. Leave it off on
`staging` and `main` so promotion PRs can use merge commits.

## 3. Commit messages

[Conventional Commits](https://www.conventionalcommits.org/): `<type>: <summary>`
in the imperative, under 72 characters.

| Type | Use for |
| :--- | :--- |
| `feat` | New pipeline capability, feature or tooling |
| `fix` | Bug fix |
| `data` | Dataset change; the commit must contain the updated `.dvc` pointer and `dvc push` must already have run |
| `exp` | A recorded experiment run on an `exp/*` branch |
| `docs` | Markdown, docstrings, comments |
| `test` | Adding or fixing tests |
| `refactor` | Code change with no behaviour change |
| `style` | Formatting only |
| `chore` | Dependencies, CI, pre-commit, repository maintenance |

## 4. Local setup

```bash
git clone https://github.com/m-hassanqureshi/california_housing_mlops.git
cd california_housing_mlops
git config pull.rebase true

uv sync                          # Python version from .python-version, packages from uv.lock
uv run pre-commit install        # hooks run on every commit

# DagsHub needs a token even for reads; this stays in the git-ignored .dvc/config.local
uv run dvc remote modify storage --local auth basic
uv run dvc remote modify storage --local user <dagshub-username>
uv run dvc remote modify storage --local password <dagshub-token>
uv run dvc pull
```

Add or upgrade dependencies with `uv add` / `uv add --dev`, never by editing
`pyproject.toml` by hand, and commit `uv.lock` in the same commit.

## 5. Before you push

1. `uv run pre-commit run` passes. The hooks block files over 1 MB, private
   keys and leaked credentials, unstripped notebooks, and ruff errors.
2. `uv run pytest` passes.
3. If you changed data or the pipeline: `uv run dvc repro`, then
   **`uv run dvc push` before `git push`**, so reviewers can `dvc pull` what
   your pointers reference. `uv run dvc status -c` should report nothing to push.
4. Notebooks are paired with jupytext (`notebooks/NN_name.ipynb` +
   `notebooks/NN_name.py`) and committed without outputs.

Never commit data, models, `.env` files, `.dvc/config.local` or any token.

## 6. Pull requests

* Fill in every section of the PR template, including before → after metrics
  for anything that touches the pipeline.
* CI must be green before review. Don't re-run a red job hoping it passes;
  find the cause.
* The reviewer checks the ML checklist in the template, not just the code. Use
  **Request changes** for anything that leaks test data into training, breaks
  reproducibility, or skips `dvc push`.
* The author resolves review threads by pushing a fix or replying with a reason.
