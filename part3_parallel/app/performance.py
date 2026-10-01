"""
Модуль performance.py — сравнение производительности и отчёт.

Сравнивает последовательную и параллельную версии обработки,
сохраняет результат в performance_report.json.
"""

import json
import os
import sys
from pathlib import Path

# Поддержка запуска и как модуля, и как скрипта
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.producer_consumer import ParallelProcessor
else:
    from .producer_consumer import ParallelProcessor


def run_comparison(
    input_dir: str | Path,
    extensions: list[str],
    output_path: str | Path = "performance_report.json",
    n_consumers: int = 2,
    n_processes: int | None = None,
) -> dict:
    """
    Сравнивает последовательную и параллельную версии обработки.

    Параметры
    ---------
    input_dir : str | Path
        Каталог с данными.
    extensions : list[str]
        Расширения файлов.
    output_path : str | Path
        Путь к JSON-отчёту.
    n_consumers : int
        Число потоков-потребителей.
    n_processes : int | None
        Число процессов (по умолчанию — cpu_count()).

    Возвращает
    ---------
    dict
        Отчёт сравнения.
    """
    input_path = Path(input_dir)
    output_path = Path(output_path)

    print("\n" + "=" * 50)
    print("  СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 50)

    # --- Последовательная версия ---
    print("\n--- Последовательная версия ---")
    processor = ParallelProcessor(
        input_dir=input_path,
        extensions=extensions,
        n_consumers=n_consumers,
        n_processes=n_processes,
    )
    seq_stats, _ = processor.run_sequential()

    # --- Параллельная версия ---
    print("\n--- Параллельная версия ---")
    par_stats, _ = processor.run()

    # --- Сравнение ---
    speedup = (
        seq_stats["elapsed_seconds"] / par_stats["elapsed_seconds"]
        if par_stats["elapsed_seconds"] > 0
        else 0
    )

    report = {
        "sequential": seq_stats,
        "parallel": par_stats,
        "speedup": round(speedup, 2),
    }

    # Сохранение отчёта
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n--- Результат ---")
    print(f"  Последовательная: {seq_stats['elapsed_seconds']}s")
    print(f"  Параллельная:     {par_stats['elapsed_seconds']}s")
    print(f"  Ускорение:        x{speedup:.2f}")
    print(f"  Отчёт сохранён:   {output_path}")

    return report


def main():
    """Точка входа для CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Сравнение производительности обработки данных"
    )
    parser.add_argument(
        "--path", "-p", type=str, required=True,
        help="Путь к каталогу с данными",
    )
    parser.add_argument(
        "--ext", "-e", type=str, default=".csv",
        help="Расширения файлов через запятую",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="performance_report.json",
        help="Путь к выходному JSON-отчёту",
    )
    parser.add_argument(
        "--consumers", "-c", type=int, default=2,
        help="Число потоков-потребителей",
    )
    parser.add_argument(
        "--processes", type=int, default=None,
        help="Число процессов (по умолчанию: cpu_count)",
    )

    args = parser.parse_args()
    extensions = [e.strip() for e in args.ext.split(",")]

    run_comparison(
        input_dir=args.path,
        extensions=extensions,
        output_path=args.output,
        n_consumers=args.consumers,
        n_processes=args.processes,
    )


if __name__ == "__main__":
    main()
