"""
Модуль scanner.py — сканирование каталогов и преобразование файлов.

Содержит функции:
- scan_directory() — рекурсивный обход дерева каталогов;
- load_file_as_array() — чтение одного файла (CSV или изображение);
- prepare_dataset() — объединение всех файлов в один .npy.
"""

import os
import time
from pathlib import Path

import numpy as np

from .config import Config
from .logging_config import setup_logger

logger = setup_logger("scanner")


def scan_directory(
    root: Path,
    extensions: list[str],
) -> list[Path]:
    """
    Рекурсивно обходит директорию и находит все файлы
    с заданными расширениями.

    Использует pathlib.Path.rglob() для обхода дерева.

    Параметры
    ---------
    root : Path
        Корневая директория для сканирования.
    extensions : list[str]
        Список расширений, например ['.csv', '.png'].

    Возвращает
    ---------
    list[Path]
        Список путей к найденным файлам.
    """
    if not root.exists():
        raise FileNotFoundError(f"Директория не найдена: {root}")

    # Нормализуем расширения: '.csv' → 'csv'
    exts_normalized = [
        e.lstrip(".").lower() for e in extensions if e.strip()
    ]

    found_files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lstrip(".").lower() in exts_normalized:
            found_files.append(path)

    logger.info("Сканирование %s: найдено %d файлов", root, len(found_files))
    return found_files


def load_file_as_array(file_path: Path) -> np.ndarray:
    """
    Читает один файл и преобразует его в массив NumPy.

    Для CSV: загружает через np.loadtxt (skiprows=1 если есть заголовок).
    Для изображений: загружает через PIL, преобразует в grayscale,
    меняет размер до IMAGE_TARGET_SIZE и нормализует.

    Параметры
    ---------
    file_path : Path
        Путь к файлу.

    Возвращает
    ---------
    np.ndarray
        Массив данных из файла.
    """
    ext = file_path.suffix.lower()

    if ext == ".csv":
        # Пытаемся пропустить заголовок, если он нечисловой
        try:
            data = np.loadtxt(file_path, delimiter=",", skiprows=0)
        except ValueError:
            data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data

    elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "Pillow не установлен. Установите: pip install Pillow"
            ) from exc

        img = Image.open(file_path).convert("L")  # grayscale
        img = img.resize(Config.IMAGE_TARGET_SIZE)
        arr = np.array(img, dtype=np.float32)
        arr /= Config.IMAGE_NORMALIZE_FACTOR  # нормализация 0–1
        return arr.flatten()  # 1D-вектор

    else:
        raise ValueError(f"Неподдерживаемое расширение: {ext}")


def prepare_dataset(
    root: Path,
    extensions: list[str],
    output_path: Path,
) -> dict:
    """
    Обходит директорию, загружает все файлы, объединяет в один массив
    и сохраняет в .npy-файл. Выводит статистику обработки.

    Параметры
    ---------
    root : Path
        Корневая директория с данными.
    extensions : list[str]
        Расширения файлов для обработки.
    output_path : Path
        Путь к выходному .npy-файлу.

    Возвращает
    ---------
    dict
        Статистика: количество файлов, размеры, время.
    """
    start_time = time.time()

    files = scan_directory(root, extensions)
    if not files:
        logger.warning("Не найдено файлов с расширениями %s", extensions)
        return {
            "files_count": 0,
            "output_shape": None,
            "elapsed_seconds": 0.0,
        }

    arrays: list[np.ndarray] = []
    errors: list[str] = []

    for fp in files:
        try:
            arr = load_file_as_array(fp)
            arrays.append(arr)
        except Exception as exc:
            errors.append(f"{fp.name}: {exc}")
            logger.error("Ошибка загрузки %s: %s", fp.name, exc)

    if not arrays:
        logger.error("Не удалось загрузить ни одного файла.")
        return {
            "files_count": 0,
            "errors": errors,
            "elapsed_seconds": time.time() - start_time,
        }

    # Объединяем все массивы
    result = np.vstack(arrays) if all(a.ndim > 1 for a in arrays) else np.array(arrays)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, result)

    elapsed = time.time() - start_time
    stats = {
        "files_found": len(files),
        "files_loaded": len(arrays),
        "files_failed": len(errors),
        "errors": errors,
        "output_shape": list(result.shape),
        "output_path": str(output_path),
        "elapsed_seconds": round(elapsed, 3),
    }

    logger.info("Готово: %d файлов загружено, shape=%s, время=%.3fs",
                len(arrays), result.shape, elapsed)

    return stats
