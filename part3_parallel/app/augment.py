"""
Модуль augment.py — функции аугментации данных.

Содержит функции для случайных преобразований:
- случайный поворот;
- горизонтальное отражение;
- добавление гауссова шума;
- случайное смещение яркости.

Аугментация выполняется на массивах NumPy, что позволяет
использовать её в многопроцессорном пуле.
"""

import numpy as np


def random_rotation(arr: np.ndarray, max_angle_deg: float = 15.0) -> np.ndarray:
    """
    Случайный поворот одномерного массива (как изображения 1D).

    Для 1D — лёгкий сдвиг значений. Для 2D — поворот матрицы.

    Параметры
    ---------
    arr : np.ndarray
        Входной массив.
    max_angle_deg : float
        Максимальный угол поворота в градусах.

    Возвращает
    ---------
    np.ndarray
        Преобразованный массив той же формы.
    """
    angle = np.random.uniform(-max_angle_deg, max_angle_deg)
    shift = int(len(arr) * angle / 90)
    if shift == 0:
        return arr.copy()
    result = np.roll(arr, shift)
    return result


def random_flip(arr: np.ndarray) -> np.ndarray:
    """
    Случайное отражение массива.

    Параметры
    ---------
    arr : np.ndarray
        Входной массив.

    Возвращает
    ---------
    np.ndarray
        Отражённый или исходный массив (50/50).
    """
    if np.random.random() > 0.5:
        return np.flip(arr).copy()
    return arr.copy()


def add_gaussian_noise(
    arr: np.ndarray,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Добавляет гауссов шум к массиву.

    Параметры
    ---------
    arr : np.ndarray
        Входной массив (значения 0–1).
    noise_std : float
        Стандартное отклонение шума.

    Возвращает
    ---------
    np.ndarray
        Массив с добавленным шумом, обрезанный до [0, 1].
    """
    noise = np.random.normal(0, noise_std, arr.shape)
    noisy = arr + noise
    return np.clip(noisy, 0.0, 1.0)


def random_brightness_shift(
    arr: np.ndarray,
    max_shift: float = 0.2,
) -> np.ndarray:
    """
    Случайное смещение яркости.

    Параметры
    ---------
    arr : np.ndarray
        Входной массив (значения 0–1).
    max_shift : float
        Максимальное смещение яркости.

    Возвращает
    ---------
    np.ndarray
        Массив со смещённой яркостью, обрезанный до [0, 1].
    """
    shift = np.random.uniform(-max_shift, max_shift)
    return np.clip(arr + shift, 0.0, 1.0)


def augment_array(arr: np.ndarray) -> np.ndarray:
    """
    Применяет случайную комбинацию аугментаций к массиву.

    Параметры
    ---------
    arr : np.ndarray
        Входной массив.

    Возвращает
    ---------
    np.ndarray
        Аугментированный массив той же формы.
    """
    result = arr.copy().astype(np.float32)

    # Каждая аугментация применяется с вероятностью 50%
    if np.random.random() > 0.5:
        result = random_rotation(result)
    if np.random.random() > 0.5:
        result = random_flip(result)
    if np.random.random() > 0.5:
        result = add_gaussian_noise(result)
    if np.random.random() > 0.5:
        result = random_brightness_shift(result)

    return result
