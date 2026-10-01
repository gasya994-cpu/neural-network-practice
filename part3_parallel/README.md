# Часть 3. Многопоточная и многопроцессорная обработка

## Описание
Producer-consumer архитектура:
- Поток-производитель сканирует директорию;
- Потоки-потребители читают файлы;
- Пул процессов применяет аугментацию (CPU-bound).

## Структура
```
part3_parallel/
├── app/
│   ├── producer_consumer.py — параллельная обработка
│   ├── augment.py           — функции аугментации
│   └── performance.py       — сравнение производительности
└── README.md
```

## Запуск
```bash
cd part3_parallel
python -m app.performance --path ../data/input --ext .csv --output performance_report.json
```
