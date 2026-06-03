"""Загрузка датасета cat / dog / panda с Kaggle и нормализация структуры.

Скачивает датасет через kagglehub и раскладывает изображения по папкам
data/cat, data/dog, data/panda (имена совпадают с Config.class_names).

Запуск:
    python scripts/download_data.py
    python scripts/download_data.py --force   # перекачать заново
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Корректный вывод кириллицы в Windows-консоли (cp1251 не кодирует часть символов).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Позволяем запускать скрипт напрямую (python scripts/download_data.py),
# добавив корень репозитория в путь импорта.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config  # noqa: E402

# Идентификатор датасета на Kaggle.
DATASET_ID = "ashishsaxena2209/animal-image-datasetdog-cat-and-panda"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_class_dirs(source_root: Path, class_names: tuple[str, ...]) -> dict[str, Path]:
    """Находим по одной папке-источнику на каждый класс.

    В архиве датасета набор изображений продублирован (animals/<class> и
    вложенная animals/animals/<class>). Чтобы не считать одни и те же картинки
    дважды и не словить утечку между train/test, для каждого класса берём только
    самую неглубокую подходящую папку (cats / dogs / panda).
    """
    candidates: dict[str, list[Path]] = {cls: [] for cls in class_names}
    for directory in source_root.rglob("*"):
        if not directory.is_dir():
            continue
        name = directory.name.lower()
        for cls in class_names:
            if name in (cls, f"{cls}s"):
                candidates[cls].append(directory)

    chosen: dict[str, Path] = {}
    for cls, dirs in candidates.items():
        if not dirs:
            raise RuntimeError(
                f"Не найдена папка для класса '{cls}' в {source_root}. "
                "Структура датасета могла измениться."
            )
        # Самая неглубокая папка = минимальное число компонентов пути.
        chosen[cls] = min(dirs, key=lambda p: len(p.parts))
    return chosen


def _gather_images(source_root: Path, class_names: tuple[str, ...]) -> dict[str, list[Path]]:
    """Группируем изображения по классам из выбранных папок-источников."""
    class_dirs = _find_class_dirs(source_root, class_names)
    grouped: dict[str, list[Path]] = {}
    for cls, directory in class_dirs.items():
        grouped[cls] = [
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return grouped


def download(cfg: Config, force: bool = False) -> None:
    """Скачиваем датасет и копируем изображения в data/<class>."""
    import kagglehub

    cfg.ensure_dirs()

    # Если данные уже на месте и не передан --force — ничего не делаем.
    existing = {
        cls: list((cfg.data_dir / cls).glob("*")) if (cfg.data_dir / cls).exists() else []
        for cls in cfg.class_names
    }
    if not force and all(existing[cls] for cls in cfg.class_names):
        print("Датасет уже загружен. Используйте --force для повторной загрузки.")
        for cls in cfg.class_names:
            print(f"  {cls}: {len(existing[cls])} изображений")
        return

    print(f"Скачиваю датасет {DATASET_ID} ...")
    source_root = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"Скачано в кеш: {source_root}")

    grouped = _gather_images(source_root, cfg.class_names)

    for cls in cfg.class_names:
        target_dir = cfg.data_dir / cls
        if force and target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        files = grouped[cls]
        if not files:
            raise RuntimeError(
                f"Не найдено изображений для класса '{cls}' в {source_root}. "
                "Структура датасета могла измениться."
            )

        for src in files:
            shutil.copy2(src, target_dir / src.name)
        print(f"  {cls}: скопировано {len(files)} изображений -> {target_dir}")

    print("Готово.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать датасет cat/dog/panda с Kaggle.")
    parser.add_argument("--force", action="store_true", help="Перекачать и перезаписать данные.")
    args = parser.parse_args()

    download(Config(), force=args.force)


if __name__ == "__main__":
    main()
