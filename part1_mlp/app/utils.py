"""
Модуль utils.py — вспомогательные функции для части 1.

Содержит функции ввода-вывода и отрисовки графика ошибки.
"""

import numpy as np


def input_float(prompt: str) -> float:
    """
    Запрашивает у пользователя число с плавающей точкой.

    Параметры
    ---------
    prompt : str
        Текст приглашения.

    Возвращает
    ---------
    float
    """
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  ! Введите корректное число (например 0.01).")


def input_int(prompt: str) -> int:
    """
    Запрашивает у пользователя целое число.

    Параметры
    ---------
    prompt : str
        Текст приглашения.

    Возвращает
    ---------
    int
    """
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("  ! Введите целое число.")


def plot_loss_history(history: list[float]) -> None:
    """
    Строит и показывает график ошибки обучения.

    Параметры
    ---------
    history : list[float]
        Список значений MSE по эпохам.
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(history) + 1), history, linewidth=1.5)
        plt.xlabel("Эпоха")
        plt.ylabel("MSE")
        plt.title("Кривая обучения")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("  matplotlib не установлен — график недоступен.")
        print(f"  История ошибок (последние 10): {history[-10:]}")


def print_predictions(
    predictions: np.ndarray,
    y_true: np.ndarray | None = None,
) -> None:
    """
    Выводит предсказания в консоль, optionally сравнивая с истинными значениями.

    Параметры
    ---------
    predictions : np.ndarray
        Массив предсказаний.
    y_true : np.ndarray | None
        Массив истинных значений (если есть).
    """
    for i, pred in enumerate(predictions):
        if y_true is not None and i < len(y_true):
            print(f"  Пример {i + 1}: предсказание = {pred[0]:.4f}, "
                  f"истинное = {y_true[i][0]:.4f}")
        else:
            print(f"  Пример {i + 1}: предсказание = {pred[0]:.4f}")
