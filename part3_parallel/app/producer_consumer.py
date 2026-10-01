"""
Модуль producer_consumer.py — параллельная обработка данных.

Архитектура producer-consumer:
- Поток-производитель сканирует директорию и кладёт пути в queue.Queue.
- Потоки-потребители читают файлы, преобразуют в массивы,
  кладут в multiprocessing.Queue.
- Пул процессов применяет аугментацию (CPU-bound) и собирает результат.

Синхронизация:
- Общий счётчик защищён threading.Lock.
- Межпроцессный обмен — через multiprocessing.Queue.
- Корректное завершение через sentinel-значения.
"""

import os
import sys
import time
import threading
import multiprocessing as mp
from pathlib import Path
from queue import Queue, Empty

import numpy as np

# Поддержка запуска и как модуля, и как скрипта
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.augment import augment_array
else:
    from .augment import augment_array


# ---------------------------------------------------------------------------
# Worker-функция для процессов (должна быть на уровне модуля)
# ---------------------------------------------------------------------------

def _augment_worker(arr: np.ndarray) -> np.ndarray:
    """
    Функция-воркер для многопроцессорной аугментации.

    Выполняется в отдельном процессе. Применяет аугментацию к массиву.

    Параметры
    ---------
    arr : np.ndarray
        Исходный массив.

    Возвращает
    ---------
    np.ndarray
        Аугментированный массив.
    """
    return augment_array(arr)


def _sequential_augment(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """
    Последовательная аугментация (для сравнения производительности).

    Параметры
    ---------
    arrays : list[np.ndarray]
        Список исходных массивов.

    Возвращает
    ---------
    list[np.ndarray]
        Список аугментированных массивов.
    """
    return [augment_array(a) for a in arrays]


# ---------------------------------------------------------------------------
# Класс ParallelProcessor
# ---------------------------------------------------------------------------

class ParallelProcessor:
    """
    Организует параллельную обработку данных по паттерну producer-consumer.

    Потоки-производители: 1 (сканирование директории).
    Потоки-потребители: N (чтение файлов).
    Процессы: M (аугментация, CPU-bound).

    Параметры
    ---------
    input_dir : Path
        Каталог с входными файлами (CSV или изображения).
    extensions : list[str]
        Расширения файлов для обработки.
    n_consumers : int
        Число потоков-потребителей (чтение файлов).
    n_processes : int | None
        Число процессов для аугментации.
    """

    SENTINEL = None  # Сигнал завершения для очередей

    def __init__(
        self,
        input_dir: Path,
        extensions: list[str],
        n_consumers: int = 2,
        n_processes: int | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.extensions = [
            e.lstrip(".").lower() for e in extensions if e.strip()
        ]
        self.n_consumers = n_consumers
        self.n_processes = n_processes or mp.cpu_count()

        # Очереди (создаются лениво при запуске параллельного режима)
        self._file_queue: Queue | None = None
        self._array_queue: mp.Queue | None = None

        # Синхронизация
        self._lock = threading.Lock()
        self._processed_count = 0
        self._error_count = 0

        # Потоки
        self._consumer_threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # Поток-производитель: сканирование директории
    # ------------------------------------------------------------------

    def _producer(self) -> None:
        """
        Сканирует директорию и кладёт пути файлов в очередь.
        Работает в отдельном потоке.
        """
        files_found = 0
        for path in self.input_dir.rglob("*"):
            if path.is_file() and path.suffix.lstrip(".").lower() in self.extensions:
                self._file_queue.put(path)
                files_found += 1

        # Sentinels для каждого потребителя
        for _ in range(self.n_consumers):
            self._file_queue.put(self.SENTINEL)

        print(f"  [Producer] Найдено файлов: {files_found}")

    # ------------------------------------------------------------------
    # Потоки-потребители: чтение файлов
    # ------------------------------------------------------------------

    def _consumer(self, consumer_id: int) -> None:
        """
        Читает файлы из очереди, преобразует в массивы,
        кладёт в multiprocessing.Queue.

        Параметры
        ---------
        consumer_id : int
            Идентификатор потока-потребителя.
        """
        while True:
            try:
                file_path = self._file_queue.get(timeout=5)
            except Empty:
                break

            if file_path is self.SENTINEL:
                self._array_queue.put(self.SENTINEL)
                break

            try:
                arr = self._load_file(file_path)
                self._array_queue.put(("data", arr))
            except Exception as exc:
                with self._lock:
                    self._error_count += 1
                print(f"  [Consumer-{consumer_id}] Ошибка {file_path.name}: {exc}")

    # ------------------------------------------------------------------
    # Загрузка одного файла
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(file_path: Path) -> np.ndarray:
        """
        Читает файл и возвращает массив NumPy.

        Параметры
        ---------
        file_path : Path
            Путь к файлу.

        Возвращает
        ---------
        np.ndarray
        """
        ext = file_path.suffix.lower()
        if ext == ".csv":
            try:
                data = np.loadtxt(file_path, delimiter=",")
            except ValueError:
                data = np.loadtxt(file_path, delimiter=",", skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return data
        elif ext in (".png", ".jpg", ".jpeg", ".bmp"):
            from PIL import Image
            img = Image.open(file_path).convert("L").resize((64, 64))
            return np.array(img, dtype=np.float32).flatten() / 255.0
        else:
            raise ValueError(f"Неподдерживаемое расширение: {ext}")

    # ------------------------------------------------------------------
    # Запуск параллельной версии
    # ------------------------------------------------------------------

    def run(self) -> tuple[dict, list[np.ndarray]]:
        """
        Запускает параллельную обработку данных.

        Возвращает
        ---------
        tuple[dict, list[np.ndarray]]
            Статистика и список аугментированных массивов.
        """
        # Инициализация очередей для параллельного режима
        self._file_queue = Queue()
        self._array_queue = mp.Queue()

        start_time = time.time()

        # Поток-производитель
        producer_thread = threading.Thread(
            target=self._producer, name="Producer", daemon=True
        )
        producer_thread.start()

        # Потоки-потребители
        for i in range(self.n_consumers):
            t = threading.Thread(
                target=self._consumer,
                args=(i,),
                name=f"Consumer-{i}",
                daemon=True,
            )
            self._consumer_threads.append(t)
            t.start()

        # Сбор массивов из multiprocessing.Queue
        arrays: list[np.ndarray] = []
        sentinels_received = 0

        while sentinels_received < self.n_consumers:
            item = self._array_queue.get()
            if item is self.SENTINEL:
                sentinels_received += 1
                continue
            _, arr = item
            arrays.append(arr)
            with self._lock:
                self._processed_count += 1

        # Ожидание завершения потоков
        producer_thread.join(timeout=5)
        for t in self._consumer_threads:
            t.join(timeout=5)

        # Многопроцессорная аугментация
        print(f"  [Main] Аугментация {len(arrays)} массивов "
              f"в {self.n_processes} процессах...")

        with mp.Pool(self.n_processes) as pool:
            augmented = pool.map(_augment_worker, arrays)

        elapsed = time.time() - start_time

        stats = {
            "files_processed": self._processed_count,
            "files_failed": self._error_count,
            "arrays_augmented": len(augmented),
            "n_consumers": self.n_consumers,
            "n_processes": self.n_processes,
            "elapsed_seconds": round(elapsed, 3),
        }

        print(f"  [Main] Готово: {self._processed_count} файлов, "
              f"время = {elapsed:.3f}s")

        return stats, augmented

    # ------------------------------------------------------------------
    # Последовательная версия (для сравнения)
    # ------------------------------------------------------------------

    def run_sequential(self) -> tuple[dict, list[np.ndarray]]:
        """
        Последовательная обработка (без потоков и процессов).

        Возвращает
        ---------
        tuple[dict, list[np.ndarray]]
            Статистика и список аугментированных массивов.
        """
        start_time = time.time()

        files = [
            p for p in self.input_dir.rglob("*")
            if p.is_file() and p.suffix.lstrip(".").lower() in self.extensions
        ]

        arrays: list[np.ndarray] = []
        for fp in files:
            try:
                arr = self._load_file(fp)
                arrays.append(arr)
            except Exception as exc:
                print(f"  [Sequential] Ошибка {fp.name}: {exc}")

        augmented = _sequential_augment(arrays)
        elapsed = time.time() - start_time

        stats = {
            "files_processed": len(arrays),
            "files_failed": len(files) - len(arrays),
            "arrays_augmented": len(augmented),
            "elapsed_seconds": round(elapsed, 3),
        }

        print(f"  [Sequential] Готово: {len(arrays)} файлов, "
              f"время = {elapsed:.3f}s")

        return stats, augmented
