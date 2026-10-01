"""
Модуль cli.py — командная строка для части 2.

Поддерживает две команды:
- prepare: сканирование и препроцессинг данных
- doctor: проверка окружения

Использует argparse. Все пути читаются из CLI-аргументов,
переменных окружения или значений по умолчанию.
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from .config import Config
from .logging_config import setup_logger
from .scanner import prepare_dataset

logger = setup_logger("cli")


# ---------------------------------------------------------------------------
# Команда: prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    """
    Обрабатывает команду prepare: сканирует директорию,
    преобразует файлы в массивы NumPy, сохраняет в .npy.
    """
    path = Path(args.path) if args.path else Config.INPUT_DIR
    extensions = args.ext.split(",") if args.ext else Config.DEFAULT_EXTENSIONS
    output = Path(args.output) if args.output else Config.OUTPUT_DIR / "dataset.npy"

    logger.info("Запуск prepare: path=%s, ext=%s, output=%s",
                path, extensions, output)

    try:
        stats = prepare_dataset(path, extensions, output)
        print("\n--- Статистика ---")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(f"Ошибка: {exc}")
    except Exception as exc:
        logger.error("Непредвиденная ошибка: %s", exc)
        print(f"Ошибка: {exc}")


# ---------------------------------------------------------------------------
# Команда: doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> None:
    """
    Проверяет окружение: версию Python, наличие библиотек,
    доступность CUDA, рабочую директорию.
    """
    print("=" * 50)
    print("  ПРОВЕРКА ОКРУЖЕНИЯ (doctor)")
    print("=" * 50)

    # --- Python ---
    print(f"  Python:          {platform.python_version()}")
    print(f"  Платформа:        {platform.platform()}")
    print(f"  Рабочая директория: {Path.cwd()}")

    # --- Проверка библиотек ---
    libraries = ["numpy", "PIL", "matplotlib"]
    for lib in libraries:
        try:
            if lib == "PIL":
                import PIL
                version = getattr(PIL, "__version__", "unknown")
            else:
                module = __import__(lib)
                version = getattr(module, "__version__", "unknown")
            print(f"  {lib:15s}  v{version}  ✓")
        except ImportError:
            print(f"  {lib:15s}  НЕ УСТАНОВЛЕН  ✗")

    # --- TensorFlow / PyTorch (опционально) ---
    for framework in ("tensorflow", "torch"):
        try:
            module = __import__(framework)
            version = getattr(module, "__version__", "unknown")
            print(f"  {framework:15s}  v{version}  ✓")
        except ImportError:
            print(f"  {framework:15s}  не установлен   (опционально)")

    # --- CUDA (через nvidia-smi) ---
    print("-" * 50)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"  CUDA / GPU:       {result.stdout.strip()}")
        else:
            print("  CUDA / GPU:       nvidia-smi вернул ошибку")
    except FileNotFoundError:
        print("  CUDA / GPU:       nvidia-smi не найден (CPU-only)")
    except subprocess.TimeoutExpired:
        print("  CUDA / GPU:       nvidia-smi превысил таймаут")
    except Exception as exc:
        print(f"  CUDA / GPU:       Ошибка: {exc}")

    # --- Проверка .env ---
    print("-" * 50)
    env_file = Path(".env")
    print(f"  .env файл:        {'найден' if env_file.exists() else 'не найден'}")
    print(f"  INPUT_DIR:        {Config.INPUT_DIR}")
    print(f"  OUTPUT_DIR:       {Config.OUTPUT_DIR}")
    print(f"  LOG_DIR:          {Config.LOG_DIR}")
    print(f"  LOG_LEVEL:        {Config.LOG_LEVEL}")

    print("=" * 50)


# ---------------------------------------------------------------------------
# Парсер аргументов
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Создаёт и возвращает парсер аргументов командной строки.

    Возвращает
    ---------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="CLI-утилита для препроцессинга данных нейросети",
        prog="cli.py",
    )
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    # --- prepare ---
    prep = subparsers.add_parser(
        "prepare", help="Сканировать директорию и сохранить .npy"
    )
    prep.add_argument(
        "--path", "-p", type=str, default=None,
        help="Путь к каталогу с данными (по умолчанию: из ENV или data/input)",
    )
    prep.add_argument(
        "--ext", "-e", type=str, default=None,
        help="Расширения файлов через запятую (например .csv,.png)",
    )
    prep.add_argument(
        "--output", "-o", type=str, default=None,
        help="Путь к выходному .npy-файлу",
    )
    prep.set_defaults(func=cmd_prepare)

    # --- doctor ---
    doc = subparsers.add_parser(
        "doctor", help="Проверка окружения"
    )
    doc.set_defaults(func=cmd_doctor)

    return parser


def main():
    """Точка входа для CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
