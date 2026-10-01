"""
Модуль main.py — консольное меню для части 1 (ООП-нейросеть).

Точка входа: python -m app.main  (из папки part1_mlp)
"""

import os
import sys
import numpy as np

# Поддержка запуска и как модуля, и как скрипта
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.models import NeuralNetwork
    from app.manager import DatasetManager
    from app.exceptions import (
        InvalidLayerSizeError,
        MismatchedDataError,
        ActivationNotFoundError,
        DatasetError,
        FileFormatError,
    )
    from app.utils import input_float, input_int, plot_loss_history, print_predictions
else:
    from .models import NeuralNetwork
    from .manager import DatasetManager
    from .exceptions import (
        InvalidLayerSizeError,
        MismatchedDataError,
        ActivationNotFoundError,
        DatasetError,
        FileFormatError,
    )
    from .utils import input_float, input_int, plot_loss_history, print_predictions


def main():
    """Главная функция — запуск консольного меню."""
    network: NeuralNetwork | None = None
    data_manager = DatasetManager()
    x_train = y_train = x_test = y_test = None

    while True:
        print("\n" + "=" * 50)
        print("  МЕНЮ — Нейронная сеть (MLP)")
        print("=" * 50)
        print("  1. Создать сеть")
        print("  2. Загрузить данные из CSV")
        print("  3. Обучить сеть")
        print("  4. Сделать предсказание")
        print("  5. Сохранить / загрузить веса")
        print("  6. Показать график ошибки")
        print("  7. Выход")
        print("=" * 50)

        choice = input("  Выберите пункт: ").strip()

        # ---- 1. Создать сеть ----
        if choice == "1":
            try:
                sizes_str = input(
                    "  Размеры слоёв через пробел (например 2 10 1): "
                ).strip()
                layer_sizes = list(map(int, sizes_str.split()))
                activation = input(
                    "  Функция активации (sigmoid / relu) [relu]: "
                ).strip().lower() or "relu"

                network = NeuralNetwork(layer_sizes, activation=activation)
                print(f"  Сеть создана: слои = {layer_sizes}, "
                      f"активация = {activation}")
            except InvalidLayerSizeError as exc:
                print(f"  ! Ошибка: {exc}")
            except ActivationNotFoundError as exc:
                print(f"  ! Ошибка: {exc}")
            except ValueError:
                print("  ! Введите целые числа через пробел.")

        # ---- 2. Загрузить данные ----
        elif choice == "2":
            csv_path = input(
                "  Путь к CSV-файлу (или Enter для dataset.csv): "
            ).strip()
            if not csv_path:
                csv_path = os.path.join(
                    os.path.dirname(__file__), "..", "dataset.csv"
                )
            try:
                x_all, y_all = data_manager.load_csv(csv_path)
                ratio = input_float("  Доля train (0.0–1.0) [0.8]: ") or 0.8
                if not 0 < ratio < 1:
                    ratio = 0.8
                x_train, y_train, x_test, y_test = data_manager.split(
                    ratio, seed=42
                )
                print(f"  Данные загружены: всего {len(x_all)}, "
                      f"train = {len(x_train)}, test = {len(x_test)}")
            except FileFormatError as exc:
                print(f"  ! Ошибка файла: {exc}")
            except DatasetError as exc:
                print(f"  ! Ошибка данных: {exc}")

        # ---- 3. Обучить сеть ----
        elif choice == "3":
            if network is None:
                print("  ! Сначала создайте сеть (пункт 1).")
                continue
            if x_train is None:
                print("  ! Сначала загрузите данные (пункт 2).")
                continue
            try:
                lr = input_float("  Скорость обучения [0.01]: ") or 0.01
                epochs = input_int("  Число эпох [200]: ") or 200
                batch = input_int("  Размер батча [16]: ") or 16

                print("\n  Обучение...")
                network.train(
                    x_train, y_train,
                    learning_rate=lr,
                    epochs=epochs,
                    batch_size=batch,
                    verbose=True,
                )

                # Оценка на тесте
                test_preds = network.predict(x_test)
                test_mse = np.mean((test_preds - y_test) ** 2)
                print(f"  Обучение завершено. MSE на test = {test_mse:.6f}")
            except MismatchedDataError as exc:
                print(f"  ! Ошибка: {exc}")

        # ---- 4. Предсказание ----
        elif choice == "4":
            if network is None:
                print("  ! Сначала создайте и обучите сеть.")
                continue
            mode = input(
                "  Ввести вручную (m) или из файла (f)? [m]: "
            ).strip().lower() or "m"

            if mode == "m":
                feat_str = input(
                    "  Признаки через пробел: "
                ).strip()
                try:
                    x_raw = np.array(
                        [list(map(float, feat_str.split()))]
                    )
                    x_norm = data_manager.normalize(x_raw)
                    preds = network.predict(x_norm)
                    print_predictions(preds)
                except ValueError:
                    print("  ! Введите числа через пробел.")
                except DatasetError as exc:
                    # Если данные не загружались — предсказываем без нормализации
                    preds = network.predict(x_raw)
                    print_predictions(preds)

            elif mode == "f":
                file_path = input("  Путь к файлу (CSV без заголовков): ").strip()
                try:
                    x_raw = np.loadtxt(file_path, delimiter=",")
                    if x_raw.ndim == 1:
                        x_raw = x_raw.reshape(1, -1)
                    try:
                        x_norm = data_manager.normalize(x_raw)
                    except DatasetError:
                        x_norm = x_raw
                    preds = network.predict(x_norm)
                    print_predictions(preds)
                except OSError as exc:
                    print(f"  ! Ошибка чтения файла: {exc}")
            else:
                print("  ! Неверный выбор.")

        # ---- 5. Сохранить / загрузить веса ----
        elif choice == "5":
            sub = input(
                "  Сохранить (s) или загрузить (l)? "
            ).strip().lower()

            if sub == "s":
                if network is None:
                    print("  ! Нет сети для сохранения.")
                    continue
                save_path = input(
                    "  Путь для сохранения [weights.json]: "
                ).strip() or "weights.json"
                network.save_weights(save_path)
                print(f"  Веса сохранены в {save_path}")

            elif sub == "l":
                load_path = input(
                    "  Путь к файлу весов [weights.json]: "
                ).strip() or "weights.json"
                try:
                    network = NeuralNetwork.load_weights(load_path)
                    print(f"  Веса загружены из {load_path}")
                    print(f"  Архитектура: {network.layer_sizes}, "
                          f"активация: {network.activation_name}")
                except FileNotFoundError:
                    print(f"  ! Файл не найден: {load_path}")
                except Exception as exc:
                    print(f"  ! Ошибка загрузки: {exc}")
            else:
                print("  ! Неверный выбор.")

        # ---- 6. График ошибки ----
        elif choice == "6":
            if nn is None or len(nn.history) == 0:
                print("Сначала обучите сеть.")
                continue

            print("\nИстория ошибок (Loss) — визуализация:")
            losses = nn.history
            max_loss = max(losses) if losses else 1.0
            bar_width = 30  # ширина полоски в символах

            # Показываем только каждую 10-ю эпоху, чтобы не засорять экран
            step = max(1, len(losses) // 15)
            
            for i in range(0, len(losses), step):
                loss = losses[i]
                # Считаем, сколько решёток нарисовать: чем меньше loss, тем короче полоска
                filled = int((max_loss - loss) / max_loss * bar_width)
                filled = max(0, min(filled, bar_width))  # защита от ошибок
                empty = bar_width - filled
                bar = "#" * filled + "-" * empty
                print(f"Epoch {i:03d}: [{bar}] {loss:.4f}")

            # И обязательно покажем последнюю эпоху отдельно
            last_loss = losses[-1]
            filled = int((max_loss - last_loss) / max_loss * bar_width)
            filled = max(0, min(filled, bar_width))
            empty = bar_width - filled
            bar = "#" * filled + "-" * empty
            print(f"\nEpoch {len(losses)-1:03d} (финал): [{bar}] {last_loss:.4f}\n")
