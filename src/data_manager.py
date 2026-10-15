"""Сохранение и загрузка параметров/данных."""
import json
from pathlib import Path

import numpy as np


class DataManager:
    """Работает с папкой results: json для параметров, npy для массивов."""

    def __init__(self, out_dir="results"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    def save_params(self, params_dict, name="params.json"):
        """Пишет словарь параметров в JSON."""
        path = self.out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params_dict, f, indent=2, ensure_ascii=False)
        return path

    def save_data(self, arr, name="densities.npy"):
        """Пишет массив (например, densities) в .npy."""
        path = self.out_dir / name
        np.save(path, np.asarray(arr))
        return path

    def load_params(self, name="params.json"):
        """Читает параметры обратно в словарь."""
        path = self.out_dir / name
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_data(self, name="densities.npy"):
        """Читает массив из .npy."""
        path = self.out_dir / name
        return np.load(path)