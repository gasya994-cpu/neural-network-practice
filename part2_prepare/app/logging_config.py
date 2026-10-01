"""
Модуль logging_config.py — настройка логирования.

Создаёт логгер, который пишет и в консоль, и в файл (logs/app.log).
"""

import logging
from pathlib import Path

from .config import Config


def setup_logger(name: str = "preprocess") -> logging.Logger:
    """
    Создаёт и настраивает логгер.

    Параметры
    ---------
    name : str
        Имя логгера.

    Возвращает
    ---------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # уже настроен

    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))

    # Формат
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Хендлер для консоли
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Хендлер для файла
    Config.ensure_dirs()
    log_file = Config.LOG_DIR / "app.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
