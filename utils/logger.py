"""
utils/logger.py
----------------
Centralized logging configuration.

Every module in the project calls `get_logger(__name__)` instead of setting
up its own handlers. This keeps log formatting consistent and makes it easy
to redirect logs (e.g. to stdout only, which is what Docker / Hugging Face
Spaces / Render expect) by changing configuration in a single place.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def _configure_root_logger(log_level: str = "INFO", log_to_file: bool = True, log_dir: Path = None):
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Always log to stdout so container platforms (Docker, HF Spaces, Render)
    # can capture logs without any extra configuration.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Optionally also persist logs to a rotating file for local development.
    if log_to_file:
        try:
            log_dir = log_dir or Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / "app.log", maxBytes=2_000_000, backupCount=3
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            # Read-only filesystems (some cloud containers) shouldn't crash
            # the whole app just because file logging isn't available.
            root.warning("File logging disabled: could not create log directory.")

    _CONFIGURED = True


def get_logger(name: str, log_level: str = "INFO", log_to_file: bool = True, log_dir: Path = None) -> logging.Logger:
    """Return a module-level logger, configuring the root logger on first use."""
    _configure_root_logger(log_level=log_level, log_to_file=log_to_file, log_dir=log_dir)
    return logging.getLogger(name)
