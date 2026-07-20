"""
core/preprocessing.py
----------------------
OpenCV-based preprocessing pipeline that turns a raw upload (photo of
handwriting, screenshot, or a canvas drawing exported as PNG) into a clean
image that gives the Pix2Tex model the best chance of a correct read.

Pipeline stages (in order):
    1. Decode          -> bytes to a numpy BGR array
    2. Grayscale        -> drop color information, keep intensity
    3. Polarity check   -> invert if the image is light-ink-on-dark-background
                          (whiteboard photos, dark-mode screenshots, inverted
                          canvas exports), since every later stage assumes
                          dark ink on a light background
    4. Noise check      -> measure whether this looks like a genuine
                          photo (grain/JPEG artifacts/uneven lighting) or a
                          clean digital render/canvas export
    5. Denoise+threshold -> applied ONLY for the "noisy photo" case (see
                          `NOISE_SIGMA_THRESHOLD`). Testing directly against
                          the model showed that denoising/binarizing an
                          already-clean image measurably *hurts*
                          recognition, so clean input now skips this stage
                          entirely rather than being "cleaned" unnecessarily.
    6. Crop             -> trim surrounding whitespace so the model sees a
                          tightly-framed equation
    7. Size safety net   -> NOT a forced resize. An earlier version of this
                          pipeline squeezed every crop into a fixed 224x224
                          padded square; that measurably hurt accuracy
                          (including on clean input) versus handing Pix2Tex
                          the naturally-cropped image at its native
                          resolution and aspect ratio -- which is what its
                          own internal, aspect-ratio-aware resizer expects.
                          `normalize_size` only steps in at genuine extremes
                          (a crop too small to read, or too large to be fast).

Each stage is a small pure function so the pipeline is easy to unit test and
easy to reorder/extend.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingError(Exception):
    """Raised when an input image cannot be decoded or processed."""


@dataclass
class PreprocessingResult:
    """Container for the output of the preprocessing pipeline."""

    pil_image: Image.Image        # Final image ready for the model (RGB, PIL)
    debug_stages: dict            # Intermediate numpy arrays, useful for debugging/UI preview
    quality_score: float          # 0..1 heuristic used as an input to confidence estimation


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes (from an upload or canvas export) into a BGR numpy array."""
    if not image_bytes:
        raise PreprocessingError("No image data received.")

    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise PreprocessingError(
            "Could not decode the image. Please upload a valid PNG/JPEG file."
        )
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Ensure the image is dark ink on a light background.

    Every later stage (adaptive threshold, "ink = pixel < 128" in
    crop_to_content, ink-ratio quality heuristic) assumes this polarity.
    Whiteboard/chalkboard photos, dark-mode screenshots, or inverted canvas
    exports violate that assumption silently: thresholding still "succeeds"
    (it just binarizes the *background* as if it were ink), the crop keeps
    almost the entire frame instead of tightening around the glyphs, and the
    model receives something close to a solid block -- which decodes as
    visual noise rather than a clean error we could catch and report.

    Detection strategy: Otsu's method finds the brightness threshold that
    best splits the image into two clusters. Whichever cluster contains more
    pixels is assumed to be the background (true for any equation image,
    since ink/strokes are always the minority of pixels). If that background
    cluster is dark, invert.

    Note this deliberately does NOT sample the image border to guess the
    background color: a photo can legitimately have a differently-colored
    panel, card, or vignette around the actual content (e.g. a dark
    rectangle inset within a lighter page), which would make the border an
    unreliable proxy for the *content region's* background. Counting pixels
    globally is robust to that case because the dominant color of the whole
    frame is still overwhelmingly determined by the (majority) background,
    wherever it sits in the frame.
    """
    threshold_value, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lighter_class_count = int(np.count_nonzero(mask))
    darker_class_count = mask.size - lighter_class_count

    if lighter_class_count >= darker_class_count:
        background_is_dark = False
    else:
        background_is_dark = True

    if background_is_dark:
        logger.debug(
            "Detected dark-background majority (otsu_threshold=%.1f); inverting polarity.",
            threshold_value,
        )
        return cv2.bitwise_not(gray)
    return gray


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Fast noise-level estimate (Immerkaer's method).

    Convolves with a Laplacian-like kernel that responds to noise but is
    insensitive to true edges/gradients, giving a rough noise standard
    deviation. Used to decide whether an input is a clean digital render
    (canvas export, typed image, screenshot) that should be left alone, or
    a genuinely noisy photo (paper grain, JPEG artifacts, uneven lighting)
    that benefits from denoising -- since testing showed denoising a
    clean image can *introduce* the very softness/artifacts it's meant to
    remove, measurably hurting recognition on input that didn't need it.
    """
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    conv = cv2.filter2D(gray.astype(np.float64), -1, kernel)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return float(sigma)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Remove noise while preserving stroke edges.

    fastNlMeansDenoising works well for photographed paper (JPEG artifacts,
    grain); a light Gaussian blur beforehand helps with camera moire.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    denoised = cv2.fastNlMeansDenoising(blurred, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised


def threshold(gray: np.ndarray) -> np.ndarray:
    """Binarize the image so strokes are solid black on a solid white background.

    Adaptive (Gaussian) thresholding copes with uneven lighting across a
    photographed page far better than a single global threshold, and we fall
    back to Otsu's method (global, noise-tolerant) if the adaptive result
    looks degenerate (e.g. almost entirely black/white, which happens on
    already-clean canvas exports).
    """
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10,
    )

    ink_ratio = 1.0 - (np.count_nonzero(adaptive) / adaptive.size)
    if ink_ratio < 0.002 or ink_ratio > 0.6:
        # Degenerate result (near blank or near solid) -> use Otsu instead.
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    return adaptive


def enhance_strokes(binary: np.ndarray) -> np.ndarray:
    """Slightly thicken thin handwritten strokes so they match the line-weight
    distribution of printed LaTeX fonts that Pix2Tex was trained on.

    Ink is black (0) on a white (255) background in `binary`.
    """
    ink_mask = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    thickened = cv2.dilate(ink_mask, kernel, iterations=1)
    return cv2.bitwise_not(thickened)


def _drop_border_connected_components(mask: np.ndarray) -> np.ndarray:
    """Remove connected components of an ink mask that touch the frame edge.

    After polarity normalization, an outer margin/vignette/panel-edge that
    happens to end up on the "ink" side of the mask will almost always
    touch the very border of the image (that's what makes it a margin
    rather than content). Genuine equation content essentially never
    reaches the literal edge of an uploaded image or canvas export. So:
    label connected components, discard any whose bounding box touches
    x=0, y=0, or the far edge, and keep only the interior ones.

    Returns an all-zero mask (signaling "nothing survived, use the
    original") if every component touched the border -- callers should
    treat that as "no filtering applied" rather than "no content".
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape[:2]

    result = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # skip label 0 (background)
        x, y, w, h, _area = stats[label_id]
        touches_border = x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height
        if not touches_border:
            result[labels == label_id] = 1

    return result


def crop_to_content(image: np.ndarray, padding: int = 30) -> np.ndarray:
    """Crop tightly around the ink so the model isn't distracted by margins.

    Works on either a grayscale image or an already-binarized one: the ink
    mask used to compute the bounding box is derived internally via Otsu's
    method rather than assuming a hard `pixel < 128` split, so this
    function no longer cares which stage produced its input.

    The bounding box is computed on a morphologically *opened* copy of that
    ink mask, not the raw mask. Sharp brightness boundaries elsewhere in the
    frame (e.g. the edge of a colored panel/card behind the actual content)
    can leave thin ringing artifacts after thresholding; left uncorrected,
    those get swept into the crop as if they were ink, expanding the box to
    include a phantom rectangle around the equation -- which the model then
    reads as spurious brackets or fraction bars. A small opening erases
    anything thinner than the kernel before we measure the bounding box,
    while the crop itself still pulls from the original (unopened) image so
    genuine thin strokes -- commas, minus signs, equals-sign bars -- are
    preserved in the output; only the *bounding box calculation* ignores
    hairline artifacts.
    """
    _, otsu_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_mask = (otsu_mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    interior_mask = _drop_border_connected_components(opened_mask)
    if np.count_nonzero(interior_mask) > 0:
        # Components touching the frame's outer edge were discarded (see
        # `_drop_border_connected_components`) -- almost always background
        # bleed (an outer margin, vignette, or the edge of a panel/card)
        # rather than genuine content, since real equations are essentially
        # never drawn flush against the very edge of an uploaded image.
        opened_mask = interior_mask

    coords = cv2.findNonZero(opened_mask)
    if coords is None:
        # The opening wiped out everything (e.g. content was itself only a
        # hairline) -- fall back to the raw mask rather than losing content.
        coords = cv2.findNonZero(ink_mask)

    if coords is None:
        # Nothing detected as "ink" at all -- return the original image
        # untouched, the caller will decide how to handle an effectively
        # blank canvas.
        return image

    x, y, w, h = cv2.boundingRect(coords)
    y0 = max(0, y - padding)
    x0 = max(0, x - padding)
    y1 = min(image.shape[0], y + h + padding)
    x1 = min(image.shape[1], x + w + padding)
    return image[y0:y1, x0:x1]


def normalize_size(cropped: np.ndarray, min_height: int = 48, max_dimension: int = 1400) -> np.ndarray:
    """Lightly clamp size at the extremes, without normalizing to a fixed shape.

    Earlier versions of this pipeline resized every crop into a fixed
    224x224 square (padded to preserve aspect ratio). Testing against the
    real model turned up a striking result: that forced resize measurably
    *hurt* recognition accuracy -- including on otherwise perfectly clean
    input -- compared to handing Pix2Tex the naturally-cropped image at its
    own resolution and aspect ratio. Pix2Tex ships with its own
    aspect-ratio-aware image resizer network (invoked internally whenever
    `resize=True`, the default), which is already tuned to the size/scale
    distribution the model was trained on; pre-resizing to an arbitrary
    fixed shape before that fights against it rather than helping.

    So: do nothing in the common case. Only step in at genuine extremes --
    a crop so small the model has too few pixels to work with, or so large
    that inference would be needlessly slow -- and even then, scale
    proportionally rather than force a specific shape.
    """
    h, w = cropped.shape[:2]
    if h == 0 or w == 0:
        return cropped

    if h < min_height:
        scale = min_height / h
    elif max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
    else:
        return cropped

    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(cropped, (new_w, new_h), interpolation=interpolation)


def estimate_quality(gray: np.ndarray) -> float:
    """Cheap heuristic (0..1) describing how "clean" the input looked.

    This feeds into the confidence score shown to the user: a blurry, very
    noisy, or nearly blank image is likely to produce an unreliable
    recognition even if the model doesn't error out.
    """
    # Sharpness proxy: variance of the Laplacian (higher = sharper edges).
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = float(np.clip(sharpness / 500.0, 0.0, 1.0))

    # Ink coverage: too little ink (blank) or too much (scribble/photo noise)
    # both suggest a poor scan. Derive the mask internally via Otsu rather
    # than assuming any particular polarity/binarization has happened yet.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_ratio = np.count_nonzero(mask) / mask.size
    if ink_ratio <= 0.001:
        coverage_score = 0.0
    else:
        # Ideal range for a clean equation crop is roughly 3%-25% ink coverage.
        coverage_score = float(np.clip(1.0 - abs(ink_ratio - 0.1) / 0.25, 0.0, 1.0))

    return float(np.clip(0.5 * sharpness_score + 0.5 * coverage_score, 0.0, 1.0))


# Noise sigma above which an input is treated as a "real photo" needing
# denoise + binarization, rather than a clean digital render/canvas export
# that should be left close to untouched. Chosen empirically: clean
# synthetic renders and canvas exports measured well under 2.0; visibly
# grainy/JPEG-compressed photos measured well above it.
NOISE_SIGMA_THRESHOLD = 1.5


def preprocess_image(image_bytes: bytes, target_width: int = 224, target_height: int = 224) -> PreprocessingResult:
    """Run the OpenCV preprocessing pipeline on raw image bytes.

    `target_width`/`target_height` are accepted for backwards compatibility
    with existing callers but no longer force a fixed output shape (see
    `normalize_size`): testing against the real model showed that forcing
    every crop into a fixed square measurably *hurt* recognition accuracy,
    including on otherwise clean input, so this pipeline now hands Pix2Tex
    a naturally-cropped, naturally-scaled image and lets its own internal,
    aspect-ratio-aware resizer do that job -- which is what it was trained
    to expect.

    Stages:
        1. Decode          -> bytes to a numpy BGR array
        2. Grayscale        -> drop color information, keep intensity
        3. Polarity check   -> invert if light-ink-on-dark-background
        4. Noise check      -> decide whether this looks like a genuine
                               photo (grain/JPEG artifacts/uneven lighting)
                               or a clean digital render/canvas export
        5. Denoise+threshold -> only applied for the "noisy photo" case;
                               skipped otherwise, since it measurably hurts
                               already-clean input
        6. Crop             -> trim to the ink's bounding box
        7. Size safety net   -> only steps in at extremes (see normalize_size)
    """
    logger.debug("Decoding uploaded image bytes (%d bytes).", len(image_bytes))
    bgr = decode_image_bytes(image_bytes)

    gray_raw = to_grayscale(bgr)
    gray = normalize_polarity(gray_raw)

    noise_sigma = estimate_noise_sigma(gray)
    is_noisy_photo = noise_sigma > NOISE_SIGMA_THRESHOLD

    if is_noisy_photo:
        denoised = denoise(gray)
        cleaned = enhance_strokes(threshold(denoised))
        logger.debug("Noise sigma=%.2f (>%.1f) -- applied denoise + threshold + stroke enhancement.", noise_sigma, NOISE_SIGMA_THRESHOLD)
    else:
        # Check background variation to detect paper/photo uploads vs digital renders
        bg_std = float(np.std(gray))
        if bg_std > 12.0:
            denoised = gray
            cleaned = enhance_strokes(threshold(gray))
            logger.debug("Paper background detected (std=%.2f) -- applied threshold + stroke enhancement.", bg_std)
        else:
            denoised = gray
            cleaned = gray
            logger.debug("Clean digital input (std=%.2f) -- skipping thresholding.", bg_std)

    cropped = crop_to_content(cleaned)
    sized = normalize_size(cropped)

    quality = estimate_quality(gray)

    # Pix2Tex expects an RGB PIL image.
    final_rgb = cv2.cvtColor(sized, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(final_rgb)

    debug_stages = {
        "grayscale_raw": gray_raw,
        "grayscale": gray,
        "denoised": denoised,
        "cleaned": cleaned,
        "cropped": cropped,
        "sized": sized,
    }

    logger.info(
        "Preprocessing complete. quality_score=%.3f noise_sigma=%.2f noisy_photo_path=%s",
        quality, noise_sigma, is_noisy_photo,
    )
    return PreprocessingResult(pil_image=pil_image, debug_stages=debug_stages, quality_score=quality)