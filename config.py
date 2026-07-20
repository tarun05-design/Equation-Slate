"""
config.py
---------
Centralized configuration for the Handwritten Math Equation Recognizer.

All tunable parameters (paths, limits, logging, model options) live here so
that the rest of the codebase never hard-codes "magic" values. Configuration
is environment-aware: values can be overridden with environment variables,
which is what you want when moving from local development to a container /
cloud deployment (Docker, Hugging Face Spaces, Render, etc.).
"""

import os
from pathlib import Path


class Config:
    """Base configuration shared by all environments."""

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent
    STATIC_DIR = BASE_DIR / "static"
    UPLOAD_DIR = STATIC_DIR / "uploads"
    LOG_DIR = BASE_DIR / "logs"
    MODEL_WEIGHTS_DIR = Path(os.environ.get("MODEL_WEIGHTS_DIR", BASE_DIR / "model_weights"))

    # ------------------------------------------------------------------
    # Flask
    # ------------------------------------------------------------------
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 8 * 1024 * 1024))  # 8 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

    # ------------------------------------------------------------------
    # Model / inference
    # ------------------------------------------------------------------
    # Pix2Tex will auto-download its pretrained checkpoint (encoder, decoder,
    # tokenizer, resizer network) into MODEL_WEIGHTS_DIR the first time it is
    # instantiated. Setting the checkpoint dir explicitly means the same path
    # can be mounted as a persistent volume / cached Docker layer so we don't
    # re-download weights on every container start.
    DEVICE = os.environ.get("DEVICE", "cpu")  # "cuda" if a GPU is available
    INFERENCE_TIMEOUT_SECONDS = int(os.environ.get("INFERENCE_TIMEOUT_SECONDS", 30))

    # Number of stochastic re-decodes used to build a self-consistency based
    # confidence estimate (see core/model_inference.py for details on why
    # this is necessary: pix2tex does not expose token probabilities through
    # its public API).
    CONFIDENCE_SAMPLES = int(os.environ.get("CONFIDENCE_SAMPLES", 1))

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    TARGET_IMAGE_HEIGHT = int(os.environ.get("TARGET_IMAGE_HEIGHT", 224))
    TARGET_IMAGE_WIDTH = int(os.environ.get("TARGET_IMAGE_WIDTH", 224))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_TO_FILE = os.environ.get("LOG_TO_FILE", "true").lower() == "true"


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    DEBUG = True
    TESTING = True


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    """Return the config class selected via the FLASK_ENV environment variable."""
    env = os.environ.get("FLASK_ENV", "production")
    return CONFIG_MAP.get(env, ProductionConfig)
