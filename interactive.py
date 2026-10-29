# Крутилки: V0, a, E. При движении ползунка симуляция пересчитывается
# и график |ψ(x, t_end)|² обновляется.
#
# Запуск: python interactive.py

import numpy as np
import matplotlib
# здесь Agg НЕ ставим — иначе окна не будет
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from src.solver import Params, SchrodingerSolver


# быстрые параметры, чтобы ползунки отзывались мгновенно
BASE = dict(
    sigma=2.0,
    x0=-10.0,
    L=40.0,
    N=512,
    dt=0.01,
    steps=1500,   # t_end = 15
)


def compute(V0: float, a: float, E: float) -> SchrodingerSolver:
    """Прогон одной симуляции и возврат готового солвера."""
    p = Params(V0=V0, a=a, E=E, **BASE)
    s = SchrodingerSolver(p, save_every=BASE["steps"])  # храним только финал
    s.run(verbose=False)
    return s


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plt.subplots_adjust(bottom=0.32)

    s = compute(1.0, 0.5, 0.5)
    line, = ax.plot(s.x, np.abs(s.psi) ** 2, lw=1.5)
    span = ax.axvspan(0, 0.5, color="red", alpha=0.2, label="барьер")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$|\psi(x,t_{end})|^2$")
    ax.set_title(f"t = {s.times[-1]:.2f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_ylim(0, float(np.abs(s.psi).max() ** 2) * 1.2)

    # ползунки
    ax_V0 = plt.axes([0.15, 0.20, 0.70, 0.03])
    ax_a  = plt.axes([0.15, 0.14, 0.70, 0.03])
    ax_E  = plt.axes([0.15, 0.08, 0.70, 0.03])
    sl_V0 = Slider(ax_V0, "V0", 0.1, 3.0, valinit=1.0)
    sl_a  = Slider(ax_a,  "a",  0.1, 2.0, valinit=0.5)
    sl_E  = Slider(ax_E,  "E",  0.1, 2.5, valinit=0.5)

    def update(_):
        V0 = sl_V0.val
        a  = sl_a.val
        E  = sl_E.val
        s = compute(V0, a, E)
        prob = np.abs(s.psi) ** 2

        line.set_ydata(prob)
        # axvspan — это Polygon, у него вершины в blended-координатах
        # (x в данных, y в [0,1]); сдвигаем только x.
        span.set_xy([(0, 0), (0, 1), (a, 1), (a, 0)])

        ax.set_ylim(0, max(prob.max() * 1.2, 1e-6))
        ax.set_title(f"t = {s.times[-1]:.2f}")
        fig.canvas.draw_idle()

    sl_V0.on_changed(update)
    sl_a.on_changed(update)
    sl_E.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()