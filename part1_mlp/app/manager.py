"""
Модуль manager.py — управление набором данных.

Содержит класс DatasetManager для загрузки CSV, нормализации
и разбиения на обучающую и тестовую выборки.
"""

import os
import numpy as np

from .exceptions import DatasetError, FileFormatError


class DatasetManager:
    """
    Менеджер набора данных: загрузка, нормализация, разбиение.

    Загружает CSV, где первая строка — заголовки, последний столбец —
    целевая переменная. Нормализует признаки методом стандартизации
    (z-score) и разбивает выборку на train/test.
    """

    def __init__(self):
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.headers: list[str] = []

    # ------------------------------------------------------------------
    # Загрузка CSV
    # ------------------------------------------------------------------

    def load_csv(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Загружает данные из CSV-файла.

        Первая строка — заголовки (пропускаются при чтении),
        последний столбец — целевая переменная.

        Параметры
        ---------
        path : str
            Путь к CSV-файлу.

        Возвращает
        ---------
        x_norm : np.ndarray, shape (m, n_features)
            Нормализованные признаки.
        y : np.ndarray, shape (m, 1)
            Целевая переменная.

        Исключения
        ---------
        FileFormatError — если файл не найден, пуст или некорректен.
        """
        if not os.path.exists(path):
            raise FileFormatError(f"Файл не найден: {path}")

        # Читаем заголовки
        with open(path, "r", encoding="utf-8") as f:
            header_line = f.readline().strip()
            self.headers = header_line.split(",") if header_line else []

        # Загружаем числовые данные (пропускаем заголовок)
        try:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        except ValueError as exc:
            raise FileFormatError(
                f"Не удалось разобрать CSV: {exc}"
            ) from exc

        if data.size == 0 or data.ndim < 2:
            raise FileFormatError("CSV пуст или содержит только один столбец.")

        x_raw = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)

        # Нормализация (стандартизация: z-score)
        self.mean = np.mean(x_raw, axis=0)
        self.std = np.std(x_raw, axis=0)
        # Защита от деления на ноль
        self.std = np.where(self.std < 1e-8, 1.0, self.std)

        x_norm = (x_raw - self.mean) / self.std

        self.x = x_norm
        self.y = y

        return x_norm, y

    # ------------------------------------------------------------------
    # Нормализация новых данных (по сохранённым параметрам)
    # ------------------------------------------------------------------

    def normalize(self, x_raw: np.ndarray) -> np.ndarray:
        """
        Нормализует новые данные по параметрам обучающей выборки.

        Параметры
        ---------
        x_raw : np.ndarray
            Сырые (ненормализованные) признаки.

        Возвращает
        ---------
        np.ndarray — нормализованные признаки.
        """
        if self.mean is None or self.std is None:
            raise DatasetError("Сначала загрузите обучающие данные (load_csv).")
        return (x_raw - self.mean) / self.std

    # ------------------------------------------------------------------
    # Разбиение на train/test
    # ------------------------------------------------------------------

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Разбивает выборку на обучающую и тестовую.

        Параметры
        ---------
        train_ratio : float
            Доля обучающей выборки (0.0–1.0).
        seed : int | None
            Seed для воспроизводимости.

        Возвращает
        ---------
        x_train, y_train, x_test, y_test : np.ndarray

        Исключения
        ---------
        DatasetError — если данные не загружены.
        """
        if self.x is None or self.y is None:
            raise DatasetError("Данные не загружены. Вызовите load_csv().")

        if not 0.0 < train_ratio < 1.0:
            raise DatasetError("train_ratio должен быть в интервале (0, 1).")

        n = self.x.shape[0]
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        train_size = int(n * train_ratio)

        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        x_train = self.x[train_idx]
        y_train = self.y[train_idx]
        x_test = self.x[test_idx]
        y_test = self.y[test_idx]

        return x_train, y_train, x_test, y_test
