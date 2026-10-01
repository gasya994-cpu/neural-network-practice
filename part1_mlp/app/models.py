"""
Модуль models.py — реализация полносвязной нейронной сети (MLP).

Содержит класс NeuralNetwork с поддержкой:
- инициализации весов (He / Xavier);
- прямого распространения (forward pass);
- обратного распространения ошибки (backpropagation);
- обучения на мини-батчах с историей ошибок;
- сохранения и загрузки весов в JSON.
"""

import json
import numpy as np

from .exceptions import (
    InvalidLayerSizeError,
    ActivationNotFoundError,
    MismatchedDataError,
)


# ---------------------------------------------------------------------------
# Функции активации и их производные
# ---------------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Сигмоида: f(x) = 1 / (1 + e^{-x})."""
    # clip для защиты от переполнения
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Производная сигмоиды: f'(x) = f(x) * (1 - f(x))."""
    s = _sigmoid(z)
    return s * (1.0 - s)


def _relu(z: np.ndarray) -> np.ndarray:
    """ReLU: f(x) = max(0, x)."""
    return np.maximum(0, z)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    """Производная ReLU: 1 если z > 0, иначе 0."""
    return (z > 0).astype(float)


# Реестр функций активации
ACTIVATION_REGISTRY = {
    "sigmoid": (_sigmoid, _sigmoid_derivative),
    "relu": (_relu, _relu_derivative),
}


# ---------------------------------------------------------------------------
# Класс NeuralNetwork
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Полносвязная нейронная сеть (многослойный перцептрон).

    Поддерживает произвольное число слоёв, выбор функции активации
    для скрытых слоёв и линейный выход (для задач регрессии).

    Параметры
    ---------
    layer_sizes : list[int]
        Список размеров слоёв, например [2, 10, 1] — 2 входа,
        10 нейронов в скрытом слое, 1 выход.
    activation : str
        Имя функции активации для скрытых слоёв: 'sigmoid' или 'relu'.
    seed : int | None
        Seed для воспроизводимости инициализации весов.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        seed: int | None = None,
    ):
        # --- Проверки ---
        if len(layer_sizes) < 2:
            raise InvalidLayerSizeError(
                "Нужно минимум 2 слоя (входной и выходной)."
            )
        if any(sz <= 0 for sz in layer_sizes):
            raise InvalidLayerSizeError(
                f"Все размеры слоёв должны быть > 0. Получено: {layer_sizes}"
            )

        if activation not in ACTIVATION_REGISTRY:
            raise ActivationNotFoundError(activation)

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self._act_fn, self._act_deriv = ACTIVATION_REGISTRY[activation]

        # Инициализация генератора случайных чисел
        rng = np.random.default_rng(seed)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.history: list[float] = []

        # Инициализация весов случайными числами
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            if activation == "relu":
                scale = np.sqrt(2.0 / fan_in)       # He initialization
            else:
                scale = np.sqrt(1.0 / fan_in)        # Xavier (упрощённо)
            w = rng.normal(0, scale, size=(fan_in, layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Прямой проход
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Прямое распространение (forward pass).

        Вычисляет активации для каждого слоя:
            Z^(i) = A^(i-1) @ W^(i) + b^(i)
            A^(i) = f(Z^(i))      — для скрытых слоёв
            A^(i) = Z^(i)          — для выходного слоя (линейный)

        Параметры
        ---------
        x : np.ndarray, shape (m, n_features)
            Входной батч.

        Возвращает
        ---------
        output : np.ndarray, shape (m, n_outputs)
            Предсказания сети.
        cache : list[dict]
            Кэш со значениями Z и A для каждого слоя
            (используется в backward).
        """
        if x.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Ожидалось {self.layer_sizes[0]} признаков, "
                f"получено {x.shape[1]}"
            )

        cache: list[dict] = []
        a = x

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            is_output = (i == len(self.weights) - 1)

            if is_output:
                # Линейная активация на выходе (для регрессии)
                a_next = z
            else:
                a_next = self._act_fn(z)

            cache.append({"z": z, "a": a})
            a = a_next

        return a, cache

    # ------------------------------------------------------------------
    # Обратный проход
    # ------------------------------------------------------------------

    def backward(
        self,
        cache: list[dict],
        y_true: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Обратное распространение ошибки (backpropagation).

        Градиенты вычисляются для функции потерь MSE:
            L = (1/m) * sum (y_pred - y_true)^2

        Для выходного (линейного) слоя:
            dZ = (2/m) * (y_pred - y_true)
        Для скрытых слоёв:
            dZ = dA * f'(Z)

        Градиенты весов и смещений:
            dW = A_prev^T @ dZ
            db = sum(dZ, axis=0)

        Параметры
        ---------
        cache : list[dict]
            Кэш из forward-прохода.
        y_true : np.ndarray, shape (m, n_outputs)
            Истинные значения.

        Возвращает
        ---------
        grads_w : list[np.ndarray]
        grads_b : list[np.ndarray]
        """
        m = y_true.shape[0]
        y_pred = cache[-1]["a_next"] if "a_next" in cache[-1] else None

        # Пересчитываем финальный выход
        output = cache[-1]["z"]  # линейный выход = z последнего слоя
        d_a = (2.0 / m) * (output - y_true)

        grads_w: list[np.ndarray] = [None] * len(self.weights)
        grads_b: list[np.ndarray] = [None] * len(self.biases)

        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                # Выходной слой — производная линейной функции = 1
                d_z = d_a
            else:
                d_z = d_a * self._act_deriv(cache[i]["z"])

            a_prev = cache[i]["a"]
            grads_w[i] = a_prev.T @ d_z
            grads_b[i] = np.sum(d_z, axis=0, keepdims=True)

            if i > 0:
                d_a = d_z @ self.weights[i].T

        return grads_w, grads_b

    # ------------------------------------------------------------------
    # Обновление весов
    # ------------------------------------------------------------------

    def _update_weights(
        self,
        grads_w: list[np.ndarray],
        grads_b: list[np.ndarray],
        learning_rate: float,
    ) -> None:
        """Градиентный спуск: W -= lr * dW, b -= lr * db."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
        batch_size: int = 16,
        verbose: bool = True,
    ) -> list[float]:
        """
        Обучение сети на мини-батчах с градиентным спуском.

        Параметры
        ---------
        x : np.ndarray, shape (m, n_features)
            Обучающая выборка.
        y : np.ndarray, shape (m, n_outputs)
            Целевые значения.
        learning_rate : float
            Скорость обучения (шаг градиентного спуска).
        epochs : int
            Число эпох (полных проходов по данным).
        batch_size : int
            Размер мини-батча.
        verbose : bool
            Если True — печатать ошибку каждые 10% эпох.

        Возвращает
        ---------
        history : list[float]
            Средняя ошибка (MSE) по эпохам.
        """
        if x.shape[1] != self.layer_sizes[0]:
            raise MismatchedDataError(
                f"Размер входа данных ({x.shape[1]}) не совпадает "
                f"с входным слоем ({self.layer_sizes[0]})"
            )

        m = x.shape[0]
        self.history = []
        print_interval = max(1, epochs // 10)

        for epoch in range(epochs):
            # Перемешивание индексов
            indices = np.random.permutation(m)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = max(1, m // batch_size)

            for start in range(0, m, batch_size):
                end = start + batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward
                predictions, cache = self.forward(x_batch)

                # Вычисление ошибки
                batch_mse = np.mean((predictions - y_batch) ** 2)
                epoch_loss += batch_mse

                # Дополняем кэш финальным выходом
                cache[-1]["a_next"] = predictions

                # Backward
                grads_w, grads_b = self.backward(cache, y_batch)

                # Обновление весов
                self._update_weights(grads_w, grads_b, learning_rate)

            avg_loss = epoch_loss / n_batches
            self.history.append(avg_loss)

            if verbose and (epoch % print_interval == 0 or epoch == epochs - 1):
                print(f"  Эпоха {epoch + 1:>4}/{epochs}  |  MSE: {avg_loss:.6f}")

        return self.history

    # ------------------------------------------------------------------
    # Предсказание
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает предсказания сети для входных данных.

        Параметры
        ---------
        x : np.ndarray, shape (m, n_features)
            Входные данные.

        Возвращает
        ---------
        np.ndarray, shape (m, n_outputs)
        """
        predictions, _ = self.forward(x)
        return predictions

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save_weights(self, path: str) -> None:
        """
        Сохраняет веса и смещения в JSON-файл.

        Параметры
        ---------
        path : str
            Путь к файлу (например, 'weights.json').
        """
        data = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_weights(cls, path: str) -> "NeuralNetwork":
        """
        Загружает сеть из JSON-файла.

        Параметры
        ---------
        path : str
            Путь к JSON-файлу с весами.

        Возвращает
        ---------
        NeuralNetwork
            Экземпляр сети с загруженными весами.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        net = cls(data["layer_sizes"], activation=data["activation"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
