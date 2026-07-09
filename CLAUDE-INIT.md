Codebase orientation for AI sessions. Posture and governance live in AGENTS.md (furnace-compiled); this file is the repo-specific map, read on demand.

## Overview

WreckSys is a sequential-recommendation demo for Fantasy & Paranormal novels: a GRU-based TensorFlow/Keras model trained on the Goodreads dataset, served via TensorFlow Serving, fronted by a Next.js app deployed to AWS EC2 (`wrecksys.com`). Originated as a WGU capstone project.

**Role in the fleet:** this repo has no `AGENTS.md`/`CLAUDE.md` and no `star.toml` — it is NOT onboarded onto the Pantheon/Forge fleet substrate. It is a standalone personal project mirrored into `Forge/`, with its own GitHub remote (`robfischer1/capstone`), its own GitHub Actions CI/CD (not Forgejo/Hemera), and its own AWS deploy target (not the fleet's Docker-host pattern). Treat it as independent — don't assume Hades/athena/vault-mcp conventions apply here.

## Architecture / module map

Two parallel Python packages exist — know which one you're touching:

| Path | Purpose | Status |
|---|---|---|
| `src/wrecksys_ai/` | The live ML package: config, data ETL, TFRecord pipeline, Keras model | Current — imports from `src/`, used for local training |
| `wrecksys/` (repo root) | Standalone extraction of the same model logic, packaged for Google Colab | Separate distribution target, not a superseded duplicate — keep both in sync if the model changes |
| `graveyard/` | Earlier implementations (bag-of-words, MovieLens example, old config wrapper) | Dead. Not imported anywhere live. Reference only. |

### `src/wrecksys_ai/` (the live ML package)

- `config.py` / `config.json` — `ConfigFile` loads `config.json` into a `FancyNamespace` (dict-like `SimpleNamespace`), then derives filesystem paths (`data_dir`, `dataset_dir`, `database`, `ratings`, `books`) relative to `PROJECT_ROOT = Path(__file__).parents[2]` (repo root, assuming this file stays at `src/wrecksys_ai/config.py`). `config.json` holds Goodreads source URLs, model hyperparameters (embedding/RNN dims, vocab size, batch size, dataset shard count), and file-naming templates.
- `io/` — data pipeline, re-exported from `io/__init__.py` as `download_source_data` (download.py), `load_ratings` (process.py), `load_datasets` (pipeline.py).
  - `download.py` — `FileManager`: streams Goodreads `.json.gz` sources via `fsspec`, converts to Feather via pyarrow.
  - `load.py` — builds TFRecord train/test datasets from raw ratings; `UserHistory`/`UserContext` namedtuples define the record shape (`context_id`, `context_rating`, `label_id`).
  - `process.py` — `format_books()` and friends: cleans the raw Goodreads book/ratings JSON into typed pandas frames, writes the SQLite `app.db` used by the frontend.
  - `pipeline.py` — `_create_dataset_from()`: parses `.tfrecord` files into `tf.data.Dataset` with a fixed `feature_description` (context of length `max_series_length`, single `label_id`).
- `model/`
  - `assets/models.py` — `WreckSys(keras.Model)`: GRU context encoder (`layers.ContextEncoder`) + book embedding encoder (`layers.BookEncoder`); scores via dot product over the full vocabulary; `serve()` returns top-K `recommendation_ids`/`recommendation_scores` via `tf.math.top_k` — this is the function traced into the SavedModel that TF Serving loads.
  - `assets/layers.py`, `losses.py`, `metrics.py`, `callbacks.py` — model building blocks (not read in depth this pass; check before modifying the model).
  - `model_maker.py` — `FunctionalModel`: `new()/load()/compile()/train_and_eval()/save()/export_as_saved_model()/export_to_tflite()`.
    **Gotcha:** the bottom of this file has module-level side effects — `test_model = FunctionalModel('test4'); test_model.new().compile().train_and_eval(...).save()...` runs on *import*, not behind `if __name__ == "__main__"`. Importing this module trains a model. Don't import it casually.
  - `model_test.py` — manual smoke-test script (loads the exported SavedModel, runs a prediction, queries `app.db` for display rows). Also has import-time execution guarded by a commented-out block — check before running.
  - `model_dir/export/` — a checked-in exported SavedModel (`saved_model.pb` + `variables/`), the artifact TF Serving mounts in `docker/serving/Dockerfile`.

### `src/wrecksys_one/` (Next.js frontend)

- `src/app/` — Next.js App Router pages: `page.js` (home), `about/`, `books/`, `books/suggest/`, `capstone/`.
- `src/app/api/predict/route.js` — POST: takes `{book_ids, book_ratings}`, pads/truncates to the last 10, calls `http://serving:8501/v1/models/wrecksys:predict`, filters out books already in history.
- `src/app/api/books/route.js` — GET: paginated book listing.
- `src/app/api/_lib/db.js` — shared SQLite helper (`getBooks({page} | {bookIds})`) against `assets/app.db`; mutually exclusive `page`/`bookIds` args (throws `TypeError` if both/neither given).
- `src/components/` — `BookGrid`, `BookRating`, `FloatingNav`, `HeroImage`, `MenuTabs`, `RexIcon`, `SummarySection`.
- `src/context/` — React context providers: `BooksContextProvider`, `RatingsContextProvider`, `WrecksContextProvider`, `ContextProviders` (composition root), `ThemeRegistry/` (MUI + Emotion SSR cache setup).
- `src/trash/` — dead frontend code (old Dockerfiles, old components/routes). Not imported.
- `assets/` — `app.db` (SQLite book metadata), `export/` (SavedModel copy for local serving), `web_model/` (TF.js model — there's commented-out client-side inference code in `predict/route.js` referencing this, currently unused; serving goes through the TF Serving container instead).

## Entry points

- **Train/export a model:** `src/wrecksys_ai/model/model_maker.py` (imports trigger training — see gotcha above; the intended entry is instantiating `FunctionalModel` and chaining its methods, not running the file as-is for anything other than the demo run at the bottom).
- **Download the dataset:** `wrecksys_ai.io.download_source_data()`, or the Colab CLI: `python -m wrecksys.data` (`wrecksys/data/__main__.py`, argparse — `-d/--datadir` or `WRECKSYS_DATA_DIR` env var; delegates to `GoodreadsData` in `wrecksys/data/datasets.py`, not yet read in depth).
- **Serve predictions:** TF Serving container (`docker/serving/Dockerfile`) exposes `/v1/models/wrecksys:predict` on :8501 — the model's `serve()` method (in `assets/models.py`) is the traced signature.
- **Frontend dev server:** `cd src/wrecksys_one && yarn dev` → `:3000`.
- **Jupyter:** `python start.py` (repo root) or `docker compose up devcontainer` → `:8888`.
- **TensorBoard:** `docker compose up tensorboard` → `:6006`, reads `./build/logs` (gitignored, generated by training).

## Build / Test / Run

No `pyproject.toml`/`package.json` at repo root — Python deps are `environment.yml` (conda), JS deps live under `src/wrecksys_one/package.json` (yarn).

```bash
# Python env
conda env create -f environment.yml && conda activate wrecksys-capstone

# Full stack via Docker (root compose.yaml)
docker compose up serving        # :8501
docker compose up tensorboard    # :6006
docker compose up devcontainer   # :8888, Jupyter

# Frontend
cd src/wrecksys_one
yarn install
yarn dev     # :3000
yarn build
yarn lint

# Frontend + serving together (its own compose file)
cd src/wrecksys_one && docker compose up
```

Tests: `tests/apiTest.js` (manual fetch script against the serving container, no assertions/runner) and `data/test/schema_test.py` (an ad hoc pyarrow schema-conversion script that runs `parse_fn()` at import time) — neither is wired into a test runner or CI. There is no pytest/jest config in the repo. Treat both as scratch scripts to read, not `run`.

CI/CD (`.github/workflows/`): `deploy.yml` triggers on push to `main` touching `src/wrecksys_one/**` — builds via `docker/build-push-action`, pushes to ECR, SSHes into an EC2 host to `docker pull`+`docker run` the new image, health-checks `localhost:3000`, and dynamically opens/closes the runner's SSH ingress on the security group around the deploy. `rollback.yml` is a `workflow_dispatch` that redeploys a given commit SHA's image. `dependabot.yml` covers GitHub Actions version bumps only (weekly). No CI step trains or tests the ML package — only the frontend is deployed automatically.

## Conventions and gotchas

- **Two model packages, one model.** Changes to `assets/models.py`/`assets/layers.py` logic in `src/wrecksys_ai/` should be mirrored into `wrecksys/model/` (root) if the Colab-facing package needs to match — check `wrecksys/model/models.py` before assuming it's already in sync.
- **Import-time side effects.** `src/wrecksys_ai/model/model_maker.py` trains and exports a model as a module-level statement. `data/test/schema_test.py` calls `parse_fn()` at import time. Don't `import` these for their functions without expecting the side effect, or read them fully before running.
- **`config.json` is mutable state, not just static config.** `ConfigFile.save()` round-trips `self.data` back to `config.json`, popping/restoring the derived `paths` object around the dump. `model_maker.train_and_eval()` mutates `model_config.train_size`/`test_size` in place — this is process-lifetime config, not persisted unless `.save()` is called.
- **TensorFlow import noise is deliberately suppressed** in `io/load.py`, `io/pipeline.py`, and `model_maker.py` via a `warnings.catch_warnings()` + logger-level dance before `import tensorflow as tf` — follow the same pattern if adding new TF-importing modules, per the inline comment ("the dumbest piece of code I've ever written").
- **`src/wrecksys_one/src/trash/`** is dead frontend code kept in-tree; don't treat it as a reference implementation without checking it's not simply stale.
- **`graveyard/`** at repo root is explicitly retired — same rule, stronger: nothing there is imported by live code.
- **No root-level Python package manifest.** There's no `pyproject.toml`/`setup.py`, so `src/wrecksys_ai` and `wrecksys` are imported via `PYTHONPATH`/notebook `sys.path` conventions, not an installed package — check how a given entry point is actually invoked (notebook cell, `python -m`, etc.) before assuming `pip install -e .` works.
- **Frontend served-model path duality.** `predict/route.js` calls the TF Serving HTTP API by default; a commented-out branch would instead load `assets/web_model/` client-side via `tf.loadGraphModel`. Only the serving-container path is live.

## Related repos

None evident from this repo's own config — no submodules, no sibling-repo references in workflows or Dockerfiles beyond the two GitHub remotes implied by CI (`robfischer1/capstone` source, AWS ECR image). Not part of the Forgejo/Pantheon fleet (no `AGENTS.md`, no `star.toml`, GitHub Actions instead of Forgejo CI, AWS instead of the fleet's Docker hosts).
