# Исследование: T(E), T(a), T(V0) и ln(T) от a.
# Все картинки кладутся в results/ с dpi=300.
#
# Запуск: python research.py
# (может занять ~минуту, гоняется 4 * 20 симуляций)

import numpy as np

from src.solver import Params, SchrodingerSolver
from src.visualizer import Visualizer


# для хорошего разрешения по энергии — широкий пакет (узкий по k)
RES = dict(
    sigma=4.0,
    x0=-15.0,
    L=60.0,
    N=1024,
    dt=0.01,
    steps=5000,
)


def numeric_T(E: float, V0: float, a: float, verbose: bool = False) -> float:
    """Прогон одной симуляции и возврат коэффициента прохождения T."""
    p = Params(V0=V0, a=a, E=E, **RES)
    # плотности нам тут не нужны, поэтому сохраняем только финал
    s = SchrodingerSolver(p, save_every=RES["steps"])
    s.run(verbose=verbose)
    T, R, P = s.balance()
    return T


def analytic_T(E: float, V0: float, a: float) -> float:
    """
    Аналитика для прямоугольного барьера:
      E < V0:  T = [1 + V0² sinh²(κa) / (4E(V0−E))]⁻¹,  κ = sqrt(2(V0−E))
      E > V0:  T = [1 + V0² sin²(ka)  / (4E(E−V0))]⁻¹,  k = sqrt(2(E−V0))
      E = V0:  предельный случай T = 1 / (1 + V0 a² / 2)
    """
    if E < V0:
        kappa = np.sqrt(2 * (V0 - E))
        return 1.0 / (1.0 + V0 ** 2 * np.sinh(kappa * a) ** 2
                      / (4 * E * (V0 - E)))
    elif E > V0:
        k = np.sqrt(2 * (E - V0))
        return 1.0 / (1.0 + V0 ** 2 * np.sin(k * a) ** 2
                      / (4 * E * (E - V0)))
    else:
        return 1.0 / (1.0 + V0 * a ** 2 / 2.0)


def main() -> None:
    # заглушка-визуализатор без сетки: у него методы T(E), T(a), T(V0)
    # не используют self.x, так что можно передать None
    vis = Visualizer(x=np.array([0.0]), barrier=None)

    # ---------- (a) T(E) ----------
    print("T(E)...")
    E_list = np.linspace(0.1, 2.0, 20)
    T_num = np.array([numeric_T(E, V0=1.0, a=0.5) for E in E_list])
    T_an = np.array([analytic_T(E, V0=1.0, a=0.5) for E in E_list])
    vis.plot_T_E(E_list, T_num, T_an, out_path="results/T_E.png")

    # ---------- (b) T(a) ----------
    print("T(a)...")
    a_list = np.linspace(0.1, 1.5, 20)
    T_a = np.array([numeric_T(E=0.5, V0=1.0, a=a) for a in a_list])
    vis.plot_T_a(a_list, T_a, out_path="results/T_a.png")

    # ---------- (c) T(V0) ----------
    print("T(V0)...")
    V0_list = np.linspace(0.5, 2.0, 20)
    T_V0 = np.array([numeric_T(E=0.5, V0=V0, a=0.5) for V0 in V0_list])
    vis.plot_T_V0(V0_list, T_V0, out_path="results/T_V0.png")

    # ---------- (d) ln(T) от a ----------
    print("ln(T) от a...")
    vis.plot_ln_T_a(a_list, T_a, out_path="results/lnT_a.png")

    print("всё, графики в results/")


if __name__ == "__main__":
    main()