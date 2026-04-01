from __future__ import annotations

import logging


def setup_logging() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logging.getLogger("torch").setLevel(logging.INFO)
    logging.getLogger("torchaudio").setLevel(logging.INFO)
    logging.getLogger("av").setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
