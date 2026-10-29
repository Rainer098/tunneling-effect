"""Сохранение и загрузка параметров/данных."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class DataManager:
    """Работает с папкой results: json для параметров, npy для массивов."""

    def __init__(self, out_dir: str = "results") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    def save_params(
        self,
        params_dict: dict[str, Any],
        name: str = "params.json",
    ) -> Path:
        """Пишет словарь параметров в JSON."""
        path = self.out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params_dict, f, indent=2, ensure_ascii=False)
        return path

    def save_data(
        self,
        arr: NDArray,
        name: str = "densities.npy",
    ) -> Path:
        """Пишет массив (например, densities) в .npy."""
        path = self.out_dir / name
        np.save(path, np.asarray(arr))
        return path

    def load_params(self, name: str = "params.json") -> dict[str, Any]:
        """Читает параметры обратно в словарь."""
        path = self.out_dir / name
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_data(self, name: str = "densities.npy") -> NDArray:
        """Читает массив из .npy."""
        path = self.out_dir / name
        return np.load(path)