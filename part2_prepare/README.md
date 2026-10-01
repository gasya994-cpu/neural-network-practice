# Часть 2. CLI-утилита для препроцессинга

## Описание
Модуль для сканирования каталога и преобразования файлов
(CSV, изображения) в единый .npy-массив.

## Команды
- `prepare --path <dir> --ext <exts> --output <file.npy>` — сканирование и сохранение
- `doctor` — проверка окружения (Python, библиотеки, CUDA)

## Структура
```
part2_prepare/
├── app/
│   ├── cli.py            — CLI-точка входа (argparse)
│   ├── config.py         — конфигурация из ENV
│   ├── scanner.py        — сканирование и загрузка
│   └── logging_config.py — логирование
└── README.md
```

## Запуск
```bash
cd part2_prepare
python -m app.cli prepare --path ../data/input --ext .csv --output ../data/output/dataset.npy
python -m app.cli doctor
```
