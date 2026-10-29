"""Прямоугольный потенциальный барьер."""
import numpy as np


class PotentialBarrier:
    """
    Барьер высоты V0 и ширины a, стоящий на отрезке 0 < x < a.

    В атомных единицах (ħ = m = 1) потенциал:
        V(x) = V0,  0 < x < a
        V(x) = 0,   иначе
    """

    def __init__(self, V0: float, a: float) -> None:
        # отрицательная высота — это уже яма, у нас другая задача
        assert V0 >= 0, "V0 не может быть отрицательным"
        assert a > 0, "ширина барьера должна быть положительной"
        self.V0 = float(V0)
        self.a = float(a)

    def value(self, x: FloatArray | float) -> FloatArray:
        """Возвращает массив V(x) для точек x (скаляр или np.array)."""
        x = np.asarray(x)
        return np.where((x > 0) & (x < self.a), self.V0, 0.0)