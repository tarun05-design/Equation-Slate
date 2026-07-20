"""
core/model_inference.py
------------------------
Defensive wrapper around pretrained Pix2Tex (LaTeX-OCR) and TrOCR-Math models.

Responsibilities:
    * Lazily load models per process.
    * Provide confidence scores and automatic fallback to TrOCR-Math when
      Pix2Tex encounters handwritten hallucinations or low confidence.
"""

import difflib
import math
import os
import random
import threading
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image, ImageEnhance

from utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """Raised when an OCR model/weights cannot be loaded."""


class InferenceError(Exception):
    """Raised when inference on a given image fails."""


@dataclass
class RecognitionResult:
    latex: str
    confidence: float
    samples: List[str]


class TrOCRMathRecognizer:
    """Singleton wrapper around fhswf/TrOCR_Math_handwritten for handwritten math formulas."""

    _instance: Optional["TrOCRMathRecognizer"] = None
    _lock = threading.Lock()

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._processor = None
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls, device: str = "cpu") -> "TrOCRMathRecognizer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(device=device)
            return cls._instance

    def _load_model(self):
        try:
            from transformers import AutoProcessor, VisionEncoderDecoderModel

            logger.info("Loading TrOCR Math handwritten model (device=%s)...", self.device)
            processor = AutoProcessor.from_pretrained("fhswf/TrOCR_Math_handwritten")
            model = VisionEncoderDecoderModel.from_pretrained("fhswf/TrOCR_Math_handwritten")

            # Positional embedding meta tensor fix for PyTorch 2.3+
            for m in model.modules():
                if hasattr(m, "weights") and getattr(m.weights, "is_meta", False):
                    shape = m.weights.shape
                    num_embeddings, embedding_dim = shape[0], shape[1]
                    half_dim = embedding_dim // 2
                    emb = math.log(10000) / (half_dim - 1)
                    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
                    emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
                    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, embedding_dim)
                    if embedding_dim % 2 == 1:
                        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
                    m.weights = torch.nn.Parameter(emb, requires_grad=False)

            model.eval()
            if self.device != "cpu":
                model.to(self.device)
            logger.info("TrOCR Math handwritten model loaded successfully.")
            return processor, model
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load TrOCR Math model.")
            raise ModelLoadError(f"Could not load TrOCR Math model: {exc}") from exc

    def recognize(self, image: Image.Image, quality_score: float = 0.5) -> RecognitionResult:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._processor, self._model = self._load_model()

        try:
            inputs = self._processor(image.convert("RGB"), return_tensors="pt")
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = self._model.generate(inputs["pixel_values"])
            latex = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            confidence = max(0.65, min(1.0, quality_score + 0.2))
            logger.info("TrOCR Math recognition complete. latex=%r", latex)
            return RecognitionResult(latex=latex, confidence=confidence, samples=[latex])
        except Exception as exc:  # noqa: BLE001
            logger.exception("TrOCR Math inference failed.")
            raise InferenceError(f"TrOCR Math inference failed: {exc}") from exc


class Pix2TexRecognizer:
    """Process-wide singleton wrapper around pix2tex.cli.LatexOCR."""

    _instance: Optional["Pix2TexRecognizer"] = None
    _lock = threading.Lock()

    def __init__(self, device: str = "cpu", weights_dir=None):
        self.device = device
        self.weights_dir = weights_dir
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls, device: str = "cpu", weights_dir=None) -> "Pix2TexRecognizer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(device=device, weights_dir=weights_dir)
            return cls._instance

    def _load_model(self):
        try:
            from pix2tex.cli import LatexOCR
        except ImportError as exc:
            raise ModelLoadError(
                "pix2tex is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        try:
            logger.info("Loading Pix2Tex pretrained model (device=%s)...", self.device)
            model = LatexOCR(None)
            logger.info("Pix2Tex model loaded successfully.")
            return model
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load Pix2Tex model.")
            raise ModelLoadError(f"Could not load Pix2Tex model: {exc}") from exc

    @property
    def model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def is_ready(self) -> bool:
        return self._model is not None

    def warm_up(self):
        """Force model loading eagerly at app startup."""
        _ = self.model
        try:
            trocr = TrOCRMathRecognizer.get_instance(device=self.device)
            if trocr._model is None:
                trocr._processor, trocr._model = trocr._load_model()
        except Exception as exc:  # noqa: BLE001
            logger.warning("TrOCR warm up skipped: %s", exc)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    @staticmethod
    def _make_variant(image: Image.Image, rng: random.Random) -> Image.Image:
        variant = image
        angle = rng.uniform(-2.5, 2.5)
        if abs(angle) > 0.01:
            variant = variant.rotate(
                angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255)
            )

        scale = rng.uniform(0.95, 1.05)
        if abs(scale - 1.0) > 0.005:
            w, h = variant.size
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized = variant.resize((new_w, new_h), Image.BICUBIC)
            canvas = Image.new("RGB", (w, h), (255, 255, 255))
            canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
            variant = canvas

        contrast_factor = rng.uniform(0.9, 1.1)
        variant = ImageEnhance.Contrast(variant).enhance(contrast_factor)
        return variant

    def recognize(
        self,
        image: Image.Image,
        num_samples: int = 1,
        quality_score: float = 0.5,
    ) -> RecognitionResult:
        if image is None:
            raise InferenceError("No image provided for recognition.")

        try:
            model = self.model
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Model unavailable: {exc}") from exc

        num_samples = max(1, num_samples)
        rng = random.Random()
        variants = [image] + [self._make_variant(image, rng) for _ in range(num_samples - 1)]

        samples: List[str] = []
        try:
            with torch.inference_mode():
                for variant in variants:
                    prediction = model(variant)
                    samples.append(prediction.strip())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pix2Tex inference failed.")
            raise InferenceError(f"Inference failed: {exc}") from exc

        if not samples or not any(samples):
            raise InferenceError("The model did not return a recognizable equation.")

        best_latex = samples[0]

        if len(samples) > 1:
            pairwise = [
                self._similarity(samples[i], samples[j])
                for i in range(len(samples))
                for j in range(i + 1, len(samples))
            ]
            agreement = sum(pairwise) / len(pairwise)
        else:
            agreement = 1.0

        confidence = max(0.0, min(1.0, 0.6 * agreement + 0.4 * quality_score))

        # ---- Method 2: Fallback on low confidence or suspicious LaTeX artifacts -----
        suspicious_artifacts = ["\\mathcal", "\\script", "\\dot", "\\ddot", "\\kappa", "\\chi", "\\Xi", "\\lambda", "ll}", "-\\!:"]
        has_suspicious_artifact = any(art in best_latex for art in suspicious_artifacts)

        if confidence < 0.55 or has_suspicious_artifact:
            logger.info(
                "Pix2Tex result suspicious or low confidence (latex=%r, confidence=%.3f). Falling back to TrOCR-Math.",
                best_latex,
                confidence,
            )
            try:
                trocr = TrOCRMathRecognizer.get_instance(device=self.device)
                return trocr.recognize(image, quality_score=quality_score)
            except Exception as exc:  # noqa: BLE001
                logger.warning("TrOCR-Math fallback failed: %s. Using Pix2Tex result.", exc)

        logger.info(
            "Recognition complete. latex=%r agreement=%.3f quality=%.3f confidence=%.3f",
            best_latex, agreement, quality_score, confidence,
        )

        return RecognitionResult(latex=best_latex, confidence=confidence, samples=samples)


def get_recognizer(device: str = "cpu", weights_dir=None) -> Pix2TexRecognizer:
    """Convenience accessor for the process-wide singleton recognizer."""
    return Pix2TexRecognizer.get_instance(device=device, weights_dir=weights_dir)