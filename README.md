# WreckSys

A sequential recommendation system for Fantasy & Paranormal novels, built on a GRU-based TensorFlow/Keras model. Originally a WGU capstone project designed to give deliberately terrible recommendations; now it mostly does so by accident.

Live demo and docs: [https://www.wrecksys.com/](https://www.wrecksys.com/)

Training data: the [Goodreads Dataset](https://mengtingwan.github.io/data/goodreads) (Fantasy & Paranormal genre slice).

## What's in this repo

- **`src/wrecksys_ai/`** — the current ML package: data download/ETL, TFRecord pipeline, and the Keras `WreckSys` model (GRU context encoder + book embedding, trained with a global softmax loss over the book vocabulary).
- **`src/wrecksys_one/`** — the Next.js 14 frontend/demo app (deployed to `wrecksys.com`). Calls a TensorFlow Serving container for predictions and reads book metadata from a bundled SQLite `app.db`.
- **`wrecksys/`** (repo root) — a standalone extraction of the model package for running in Google Colab, packaged separately from `src/wrecksys_ai/` so it can be pip-installed/imported without the rest of the repo.
- **`docker/`** — Dockerfiles for the TF Serving container, TensorBoard, and a Jupyter devcontainer.
- **`docs/notebooks/`** — exploratory/analysis notebooks (`source.ipynb`, `features.ipynb`, `analysis.ipynb`).
- **`graveyard/`** — retired experiments and earlier implementations, kept for reference. Not maintained, not imported by anything live.

## Installation

Compatible with Linux and Windows; on Windows, WSL is preferred for GPU (`tensorflow-gpu`) access.

Fastest path: install [Conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe), clone the repo, and run `start.py` (launches Jupyter Notebook):

```bash
git clone https://github.com/robfischer1/capstone wrecksys
cd wrecksys
conda env create -f environment.yml
conda activate wrecksys-capstone
python start.py
```

Manual dependency install: see `environment.yml` at the repo root (Python 3.11, pandas, pyarrow, TensorFlow/GPU, Node/Yarn for the frontend).

## Running the full stack (Docker)

The root `compose.yaml` builds three services from `docker/`:

```bash
docker compose up serving        # TF Serving, model API on :8501
docker compose up tensorboard     # TensorBoard on :6006, reads ./build/logs
docker compose up devcontainer    # Jupyter on :8888, mounts ./data and ./src
```

The frontend has its own compose file (`src/wrecksys_one/compose.yaml`) that pairs a `tensorflow/serving` container with the Next.js app (`robfischer1/wrecksys:demo`) on port 3000, talking to serving over the compose network on 8501.

## Frontend (`src/wrecksys_one/`)

Next.js 14 + MUI app. Key routes:

- `src/app/api/predict/route.js` — takes a reading history, calls the serving container's `/v1/models/wrecksys:predict` endpoint, filters out already-seen books.
- `src/app/api/books/route.js` — paginated book listing from `assets/app.db`.
- `src/app/api/_lib/db.js` — SQLite query helper shared by both routes.

```bash
cd src/wrecksys_one
yarn install
yarn dev      # http://localhost:3000
yarn build
yarn lint
```

Deployment: pushes to `main` touching `src/wrecksys_one/**` trigger `.github/workflows/deploy.yml` — builds the image, pushes to ECR, SSHes into an EC2 host to pull and restart the container, then health-checks it. `.github/workflows/rollback.yml` redeploys a prior image tag by commit SHA via `workflow_dispatch`.

## Development (ML package)

No formal test runner is wired into CI. `tests/apiTest.js` and `data/test/schema_test.py` are ad hoc scripts, not a pytest/jest suite — run them by hand if you need them.

```bash
python -m wrecksys_ai.model.model_maker   # trains + exports a model on import (see CLAUDE-INIT.md)
python -m wrecksys.data                   # Colab-package CLI: downloads the Goodreads dataset
```

## License

[BSD-3-Clause-Clear](https://choosealicense.com/licenses/bsd-3-clause-clear/) — Copyright (c) 2023 Rob Fischer.
