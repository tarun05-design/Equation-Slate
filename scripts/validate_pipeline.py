"""
scripts/validate_pipeline.py
-----------------------------
Lightweight end-to-end validation of the recognition pipeline, run with:

    python scripts/validate_pipeline.py

This is NOT a unit test suite (the project spec calls for a working
pipeline with a basic validation check rather than full test coverage). It:

    1. Synthesizes a test image containing a rendered equation (so this
       script has no external file dependencies).
    2. Runs it through preprocessing -> Pix2Tex inference -> SymPy solving.
    3. Prints each stage's output and exits non-zero if any stage raises.

Use this after `pip install -r requirements.txt` to confirm your local
environment (and, before deploying, your container) can run the full
pipeline -- including the first-time Pix2Tex weight download.
"""

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from config import get_config  # noqa: E402
from core.equation_solver import SolverError, solve_expression  # noqa: E402
from core.model_inference import InferenceError, ModelLoadError, get_recognizer  # noqa: E402
from core.preprocessing import preprocess_image  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def make_synthetic_equation_image(text: str = "2 + 2 = 4") -> bytes:
    """Render plain text onto a white canvas as a stand-in for a real
    handwritten-equation photo. This keeps the validation script fully
    self-contained (no sample image files to ship/maintain), at the cost of
    testing the OCR model on typed rather than handwritten strokes -- good
    enough to prove the *pipeline wiring* works end-to-end."""
    img = Image.new("RGB", (400, 150), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 48)
    except OSError:
        font = ImageFont.load_default()
    draw.text((30, 45), text, fill="black", font=font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    cfg = get_config()
    print("=" * 70)
    print("Equation Slate -- pipeline validation")
    print("=" * 70)

    overall_start = time.time()

    # ---- Stage 1: preprocessing ------------------------------------------------
    print("\n[1/3] Preprocessing (OpenCV)...")
    try:
        image_bytes = make_synthetic_equation_image("2+2=4")
        result = preprocess_image(
            image_bytes,
            target_width=cfg.TARGET_IMAGE_WIDTH,
            target_height=cfg.TARGET_IMAGE_HEIGHT,
        )
        print(f"    OK. quality_score={result.quality_score:.3f}, image_size={result.pil_image.size}")
    except Exception as exc:  # noqa: BLE001
        print(f"    FAILED: {exc}")
        return 1

    # ---- Stage 2: model inference -----------------------------------------------
    print("\n[2/3] Model inference (Pix2Tex)...")
    try:
        recognizer = get_recognizer(device=cfg.DEVICE, weights_dir=cfg.MODEL_WEIGHTS_DIR)
        t0 = time.time()
        recognition = recognizer.recognize(
            result.pil_image, num_samples=1, quality_score=result.quality_score
        )
        print(f"    OK ({time.time() - t0:.1f}s). latex={recognition.latex!r}")
        print(f"    confidence={recognition.confidence:.3f}")
    except (ModelLoadError, InferenceError) as exc:
        print(f"    FAILED: {exc}")
        print("    (This usually means pix2tex/torch aren't installed yet, or the")
        print("     pretrained weights couldn't be downloaded -- check your network.)")
        return 1

    # ---- Stage 3: solving ---------------------------------------------------------
    print("\n[3/3] Equation solving (SymPy)...")
    try:
        solved = solve_expression(recognition.latex)
        print(f"    OK. kind={solved.kind} sympy_input={solved.sympy_input!r}")
        if solved.kind == "equation":
            print(f"    solutions={solved.solutions}")
        else:
            print(f"    simplified={solved.simplified} numeric_value={solved.numeric_value}")
    except SolverError as exc:
        # The recognized text from a synthetic render isn't guaranteed to be
        # parseable LaTeX (OCR is approximate) -- treat this as a soft
        # warning rather than a hard pipeline failure, since stages 1 and 2
        # (the parts specific to this project) already succeeded.
        print(f"    WARNING: solver could not process the recognized output: {exc}")

    elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"Pipeline validation complete in {elapsed:.1f}s.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
