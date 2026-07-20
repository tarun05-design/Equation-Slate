# Equation Slate — Handwritten Math Equation Recognition & Solver

An end-to-end deep learning web application that recognizes handwritten
mathematical equations (uploaded photo or drawn on an in-browser canvas),
converts them to LaTeX with a pretrained **Pix2Tex** model, solves/simplifies
them with **SymPy**, and displays the recognized equation, LaTeX source,
solved result, and an estimated confidence score.

```
Draw / upload  ->  OpenCV preprocessing  ->  Pix2Tex (LaTeX-OCR)  ->  SymPy solve  ->  JSON  ->  UI
```

## Contents

- [Architecture](#architecture)
- [Local setup](#local-setup)
- [Running the app](#running-the-app)
- [Validating the pipeline](#validating-the-pipeline)
- [API reference](#api-reference)
- [Docker](#docker)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)

## Architecture

```
├── app.py                     # Flask app factory + routes
├── config.py                  # Environment-aware configuration
├── core/
│   ├── preprocessing.py       # OpenCV pipeline (grayscale, denoise, threshold, crop, resize)
│   ├── model_inference.py     # Pix2Tex wrapper: lazy load, inference, confidence estimation
│   └── equation_solver.py     # LaTeX -> SymPy parsing + solving/simplifying
├── utils/
│   └── logger.py              # Centralized logging configuration
├── templates/index.html       # Single-page UI (draw / upload tabs + results panel)
├── static/css/style.css       # UI styling
├── static/js/app.js           # Canvas drawing, upload, fetch() calls, result rendering
├── scripts/validate_pipeline.py  # Self-contained end-to-end pipeline check
├── requirements.txt
├── Dockerfile
└── .dockerignore / .gitignore
```

The three `core/` modules are independent and import-safe on their own —
`preprocessing.py` only depends on OpenCV/NumPy/Pillow, `equation_solver.py`
only on SymPy, and `model_inference.py` defers its `pix2tex`/`torch` imports
until the model is actually first used. This means `app.py` can be imported
and its non-model routes exercised even before the heavy ML dependencies are
installed, which keeps local iteration fast.

### Why a "confidence score" if Pix2Tex doesn't expose one?

Pix2Tex's public `LatexOCR.__call__` API returns only the decoded LaTeX
string — it does not expose per-token probabilities. Rather than invent a
number, this app computes a **self-consistency confidence**: the same
preprocessed image is decoded multiple times (`CONFIDENCE_SAMPLES`, default
3) and the agreement between those samples is measured (via sequence
similarity) and blended with an image-quality heuristic from the
preprocessing stage (sharpness + ink coverage). High agreement across
repeated decodes + a clean input image → high confidence. This is clearly
an estimate, not a calibrated probability, and is documented as such in the
code (`core/model_inference.py`).

## Local setup

Requires Python 3.10+.

```bash
git clone <your-fork-url>
cd mathrecognizer

python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **First run note:** the first time `LatexOCR()` is instantiated, Pix2Tex
> downloads its pretrained checkpoint (~115 MB total: encoder/decoder
> weights + an image-resizing network) to a local cache directory. This
> requires an internet connection once; afterwards it's cached and loads
> fully offline. See [Configuration](#configuration) for pinning the cache
> location.

## Running the app

```bash
export FLASK_ENV=development
python app.py
```

The app serves at `http://localhost:7860`. Open it in a browser, either
draw an equation on the canvas or upload an image, and click
**Recognize & Solve**.

For a production-style run locally (matches what the Dockerfile does):

```bash
gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 4 --timeout 120 app:app
```

## Validating the pipeline

A self-contained script exercises all three pipeline stages
(preprocessing → inference → solving) against a synthetically generated
test image, so it has no external file dependencies:

```bash
python scripts/validate_pipeline.py
```

Expected output ends with `Pipeline validation complete in ...s.` and prints
the intermediate output of each stage. This is the "basic end-to-end
validation" referenced in the project spec — it's a smoke test confirming
the pipeline is wired correctly, not a substitute for trying real
handwriting samples through the UI. (Pix2Tex is trained on handwritten
strokes, so a synthetically-rendered typed-font test image is a weaker
input for it than actual handwriting — see
[Known limitations](#known-limitations).)

## API reference

### `POST /api/recognize`

Accepts **either**:
- `multipart/form-data` with a file field named `image`, or
- `application/json` with `{"image_data": "data:image/png;base64,..."}` (what
  the drawing canvas sends).

Response `200`:
```json
{
  "request_id": "5003c49e",
  "latex": "x^{2}-4=0",
  "confidence": 0.87,
  "samples": ["x^{2}-4=0", "x^{2}-4=0", "x^2-4=0"],
  "solution": {
    "kind": "equation",
    "sympy_input": "x^{2}-4=0",
    "solutions": ["-2", "2"],
    "simplified": "x**2 - 4",
    "numeric_value": null
  },
  "solve_warning": null,
  "elapsed_ms": 812
}
```

Error responses (`400`/`422`/`503`) return `{"error": "...", "stage": "..."}`
where `stage` is one of `input`, `preprocessing`, `inference`, `model_load`.

### `GET /healthz`

Readiness probe used by container orchestrators: `{"status": "ok", "model_loaded": true}`.

## Docker

```bash
docker build -t equation-slate .
docker run --rm -p 7860:7860 equation-slate
```

Then visit `http://localhost:7860`. The image installs system libraries
OpenCV needs at runtime (`libgl1`, `libglib2.0-0`), runs a single Gunicorn
worker with multiple threads (see the comment in `Dockerfile` on why a
single *process* is used for a memory-heavy model like this), and exposes a
`HEALTHCHECK` against `/healthz`.

To avoid re-downloading Pix2Tex's weights on every container start, mount a
volume at the cache directory:

```bash
docker run --rm -p 7860:7860 -v equation-slate-weights:/app/model_weights equation-slate
```

## Deployment

### Option A — Hugging Face Spaces (recommended)

1. Create a new Space at huggingface.co/new-space, SDK = **Docker**.
2. Push this repository to the Space's git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
3. Spaces builds the `Dockerfile` automatically and exposes port `7860`
   (already the default in this project). No further configuration needed.
4. First build will take a few minutes (torch + pix2tex install, plus the
   first weight download on cold start). Subsequent restarts reuse the
   Space's persistent storage if you've enabled it.

### Option B — Render

1. Push this repository to GitHub.
2. On Render: **New → Web Service**, connect the repo, environment =
   **Docker**.
3. Render sets `$PORT` automatically; the Dockerfile's `CMD` already binds
   to `${PORT:-7860}`, so no changes are required.
4. Add a persistent disk mounted at `/app/model_weights` (Render → Disks) if
   you want to avoid re-downloading Pix2Tex weights on every deploy.

### Sharing your work

Once deployed, share the live Space/Render URL alongside this repository's
GitHub URL — the two together are the full deliverable (see the project
brief's Deployment section).

## Configuration

All configuration is environment-variable driven (`config.py`), so the same
code runs unmodified locally and in a container:

| Variable                 | Default              | Purpose                                             |
|---------------------------|----------------------|------------------------------------------------------|
| `FLASK_ENV`               | `production`         | `development` / `production` / `testing`             |
| `DEVICE`                  | `cpu`                | `cuda` if deploying with a GPU                        |
| `MODEL_WEIGHTS_DIR`       | `./model_weights`    | Where Pix2Tex's cache directory is created            |
| `CONFIDENCE_SAMPLES`      | `3`                  | Re-decodes used for the self-consistency confidence   |
| `TARGET_IMAGE_WIDTH/HEIGHT` | `224`               | Preprocessing output canvas size                       |
| `MAX_CONTENT_LENGTH`      | `8388608` (8 MB)     | Max upload size                                        |
| `LOG_LEVEL`               | `INFO`               | Python logging level                                   |
| `LOG_TO_FILE`             | `true`               | Also write rotating logs to `logs/app.log`             |

## Known limitations

- **Pix2Tex is trained primarily on rendered/typeset LaTeX**, not photographed
  handwriting. It handles clean, well-segmented handwriting reasonably well,
  but is the single biggest source of recognition error in this pipeline —
  messy handwriting, multi-line equations, and unusual symbols reduce
  accuracy. This is a model limitation, not a bug in the surrounding
  application.
- **Confidence is a heuristic**, not a calibrated probability (see above).
  Treat it as a relative signal ("this one looks shakier than that one"),
  not an exact error rate.
- **SymPy's `parse_latex`** covers a broad but not exhaustive subset of
  LaTeX math syntax; some valid LaTeX from Pix2Tex (unusual macros, multi-line
  systems of equations, matrices) will fail to parse. The API surfaces this
  as a `solve_warning` rather than failing the whole request, since the
  recognition step still succeeded.
- Solving is symbolic (`sympy.solve` / `sympy.simplify`); equations with no
  closed-form solution return an empty solution list rather than a numeric
  approximation.
