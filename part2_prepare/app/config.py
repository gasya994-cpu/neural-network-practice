"""
Модуль config.py — конфигурация из переменных окружения.

Все настройки читаются из .env / переменных окружения или берутся
из значений по умолчанию. Никаких захардкоженных путей.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен — работаем с системными env


class Config:
    """Конфигурация приложения части 2."""

    # Рабочая директория по умолчанию — текущая
    BASE_DIR = Path(os.getenv("BASE_DIR", Path.cwd()))

    # Каталог с входными данными
    INPUT_DIR = Path(os.getenv("INPUT_DIR", BASE_DIR / "data" / "input"))

    # Каталог для выходных файлов
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "data" / "output"))

    # Каталог для логов
    LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs"))

    # Уровень логирования
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Расширения по умолчанию
    DEFAULT_EXTENSIONS = os.getenv(
        "DEFAULT_EXTENSIONS", ".csv,.png,.jpg,.jpeg"
    ).split(",")

    # Размер батча для загрузки изображений
    IMAGE_TARGET_SIZE = tuple(
        int(s) for s in os.getenv("IMAGE_TARGET_SIZE", "64,64").split(",")
    )

    # Нормализация изображений (делитель)
    IMAGE_NORMALIZE_FACTOR = float(os.getenv("IMAGE_NORMALIZE_FACTOR", "255"))

    @classmethod
    def ensure_dirs(cls) -> None:
        """Создаёт необходимые директории, если их нет."""
        for d in (cls.INPUT_DIR, cls.OUTPUT_DIR, cls.LOG_DIR):
            d.mkdir(parents=True, exist_ok=True)
