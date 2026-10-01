"""
Кастомные исключения для нейронной сети.

Каждый класс отвечает за конкретный тип ошибки, что упрощает
диагностику и обработку в вызывающем коде.
"""


class NeuralNetworkError(Exception):
    """Базовый класс для всех исключений нейросети."""

    def __init__(self, message: str = "Ошибка нейронной сети"):
        super().__init__(message)


class InvalidLayerSizeError(NeuralNetworkError):
    """Возникает, когда размер слоя <= 0 или число слоёв < 2."""

    def __init__(self, message: str = "Некорректный размер слоя"):
        super().__init__(message)


class MismatchedDataError(NeuralNetworkError):
    """Возникает, когда размерность данных не совпадает с входным слоем."""

    def __init__(self, message: str = "Размерность данных не совпадает с сетью"):
        super().__init__(message)


class ActivationNotFoundError(NeuralNetworkError):
    """Возникает, когда запрошена неизвестная функция активации."""

    def __init__(self, name: str = ""):
        msg = f"Неизвестная функция активации: {name}" if name else "Функция активации не найдена"
        super().__init__(msg)


class DatasetError(Exception):
    """Базовый класс для ошибок DatasetManager."""

    def __init__(self, message: str = "Ошибка работы с данными"):
        super().__init__(message)


class FileFormatError(DatasetError):
    """Возникает, если файл не найден, пуст или имеет неверный формат."""

    def __init__(self, message: str = "Ошибка формата файла"):
        super().__init__(message)
