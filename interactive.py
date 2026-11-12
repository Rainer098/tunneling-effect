# Крутилки: V0, a, E. При движении ползунка симуляция пересчитывается
# и график |ψ(x, t_end)|² обновляется.
#
# Запуск: python interactive.py

"""Интерактивный скрипт. Позволяет менять параметры ползунками."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src.solver import Params, SchrodingerSolver

# Параметры для быстрой работы интерактива
BASE = dict(sigma=2.0, x0=-10.0, L=40.0, N=512, dt=0.01, steps=1500)

def compute(V0, a, E):
    """Запускает симуляцию и возвращает готовый объект Solver."""
    p = Params(V0=V0, a=a, E=E, **BASE)
    s = SchrodingerSolver(p, save_every=BASE["steps"])
    s.run(verbose=False)
    return s

def main():
    """Главная функция интерактива."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plt.subplots_adjust(bottom=0.35)

    s = compute(1.0, 0.5, 0.5)
    prob = np.abs(s.psi) ** 2

    # Создаем линию и полосу барьера ОДИН раз
    line, = ax.plot(s.x, prob, lw=1.5)
    span = ax.axvspan(0, 0.5, color="red", alpha=0.2, label="Барьер")

    ax.set_xlabel("Координата x (атомные единицы)")
    ax.set_ylabel(r"Плотность вероятности \(|\psi(x,t)|^2\)")
    ax.set_title(f"Интерактивная симуляция (t = {s.times[-1]:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_ylim(0, float(np.abs(s.psi).max() ** 2) * 1.2)

    # Слайдеры
    ax_V0 = plt.axes([0.15, 0.20, 0.70, 0.03])
    ax_a  = plt.axes([0.15, 0.13, 0.70, 0.03])
    ax_E  = plt.axes([0.15, 0.06, 0.70, 0.03])

    sl_V0 = Slider(ax_V0, "V0", 0.1, 3.0, valinit=1.0, valstep=0.05)
    sl_a  = Slider(ax_a,  "a",  0.1, 2.0, valinit=0.5, valstep=0.05)
    sl_E  = Slider(ax_E,  "E",  0.1, 2.5, valinit=0.5, valstep=0.05)

    def update(_):
        """Функция обновления графика при движении ползунка."""
        V0 = sl_V0.val
        a  = sl_a.val
        E  = sl_E.val
        s = compute(V0, a, E)
        prob = np.abs(s.psi) ** 2

        line.set_data(s.x, prob)

        # Исправление ошибки: вместо set_xy используем set_x и set_width
        span.set_x(0)
        span.set_width(a)

        ax.set_ylim(0, max(prob.max() * 1.2, 1e-6))
        ax.set_title(f"Интерактивная симуляция (t = {s.times[-1]:.2f})")
        fig.canvas.draw_idle()

    # Привязка событий
    sl_V0.on_changed(update)
    sl_a.on_changed(update)
    sl_E.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()