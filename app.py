"""
app.py
------
Flask application entry point.

Routes:
    GET  /                 -> main UI (upload + drawing canvas)
    POST /api/recognize    -> run the full pipeline on an uploaded image or
                               a canvas drawing (base64 PNG) and return JSON
    GET  /healthz          -> lightweight health/readiness check for
                               container orchestrators (Docker, HF Spaces,
                               Render)

The route handlers themselves stay thin: all real work is delegated to the
`core` package (preprocessing, model inference, equation solving) so each
concern can be tested and reasoned about independently.
"""

import base64
import binascii
import time
import uuid

from flask import Flask, jsonify, render_template, request

from config import get_config
from core.equation_solver import SolverError, solve_expression
from core.model_inference import InferenceError, ModelLoadError, get_recognizer
from core.preprocessing import PreprocessingError, preprocess_image
from utils.logger import get_logger

logger = get_logger(__name__)


def create_app():
    cfg = get_config()
    app = Flask(__name__)
    app.config.from_object(cfg)

    cfg.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    recognizer = get_recognizer(device=cfg.DEVICE, weights_dir=cfg.MODEL_WEIGHTS_DIR)
    try:
        recognizer.warm_up()
    except Exception as exc:
        logger.warning("Eager model warmup skipped: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _allowed_file(filename: str) -> bool:
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in cfg.ALLOWED_EXTENSIONS
        )

    def _extract_image_bytes(req) -> bytes:
        """Pull raw image bytes out of either a multipart file upload or a
        JSON payload containing a base64 data URL from the drawing canvas."""
        if "image" in req.files and req.files["image"].filename:
            file = req.files["image"]
            if not _allowed_file(file.filename):
                raise PreprocessingError(
                    f"Unsupported file type. Allowed: {', '.join(sorted(cfg.ALLOWED_EXTENSIONS))}"
                )
            return file.read()

        payload = req.get_json(silent=True) or {}
        data_url = payload.get("image_data")
        if not data_url:
            raise PreprocessingError("No image file or canvas drawing was provided.")

        try:
            header, encoded = data_url.split(",", 1) if "," in data_url else ("", data_url)
            return base64.b64decode(encoded)
        except (binascii.Error, ValueError) as exc:
            raise PreprocessingError("Could not decode the canvas drawing data.") from exc

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/healthz")
    def healthz():
        return jsonify(
            status="ok",
            model_loaded=recognizer.is_ready(),
        )

    @app.route("/api/recognize", methods=["POST"])
    def recognize():
        request_id = uuid.uuid4().hex[:8]
        start = time.time()
        logger.info("[%s] Incoming recognition request.", request_id)

        # ---- 1. Read input -------------------------------------------------
        try:
            image_bytes = _extract_image_bytes(request)
        except PreprocessingError as exc:
            logger.info("[%s] Bad request: %s", request_id, exc)
            return jsonify(error=str(exc), stage="input"), 400

        # ---- 2. Preprocess ---------------------------------------------------
        try:
            preprocessing_result = preprocess_image(
                image_bytes,
                target_width=cfg.TARGET_IMAGE_WIDTH,
                target_height=cfg.TARGET_IMAGE_HEIGHT,
            )
        except PreprocessingError as exc:
            logger.info("[%s] Preprocessing failed: %s", request_id, exc)
            return jsonify(error=str(exc), stage="preprocessing"), 400
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] Unexpected preprocessing error.", request_id)
            return jsonify(error="Unexpected error while processing the image.", stage="preprocessing"), 500

        # ---- 3. Recognize (Pix2Tex) ------------------------------------------
        try:
            recognition = recognizer.recognize(
                preprocessing_result.pil_image,
                num_samples=cfg.CONFIDENCE_SAMPLES,
                quality_score=preprocessing_result.quality_score,
            )
        except ModelLoadError as exc:
            logger.error("[%s] Model failed to load: %s", request_id, exc)
            return jsonify(
                error="The recognition model is unavailable right now. Please try again shortly.",
                stage="model_load",
            ), 503
        except InferenceError as exc:
            logger.info("[%s] Inference failed: %s", request_id, exc)
            return jsonify(error=str(exc), stage="inference"), 422
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] Unexpected inference error.", request_id)
            return jsonify(error="Unexpected error during recognition.", stage="inference"), 500

        # ---- 4. Solve with SymPy ----------------------------------------------
        solve_payload = None
        solve_warning = None
        try:
            solved = solve_expression(recognition.latex)
            solve_payload = {
                "kind": solved.kind,
                "sympy_input": solved.sympy_input,
                "solutions": solved.solutions,
                "simplified": solved.simplified,
                "numeric_value": solved.numeric_value,
                "steps": solved.steps,
            }
        except SolverError as exc:
            # A recognized-but-unsolvable equation is not a failure of the
            # overall pipeline -- we still return the recognition result and
            # explain why solving didn't work.
            logger.info("[%s] Solving failed: %s", request_id, exc)
            solve_warning = str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] Unexpected solver error.", request_id)
            solve_warning = "Unexpected error while solving the expression."

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info("[%s] Request complete in %d ms.", request_id, elapsed_ms)

        return jsonify(
            request_id=request_id,
            latex=recognition.latex,
            confidence=round(recognition.confidence, 4),
            samples=recognition.samples,
            solution=solve_payload,
            solve_warning=solve_warning,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------
    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify(error="Image is too large. Please upload a smaller file (max 8 MB)."), 413

    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify(error="Not found."), 404

    @app.errorhandler(500)
    def server_error(_exc):
        logger.exception("Unhandled server error.")
        return jsonify(error="Internal server error."), 500

    return app


app = create_app()


if __name__ == "__main__":
    # Local development entry point. In production, use gunicorn (see
    # Dockerfile / README) rather than the Flask dev server.
    app.run(host="0.0.0.0", port=7860, debug=app.config.get("DEBUG", False))
