"""Модуль визуализации. Строит графики с выбором формата (запятая или точка)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, PillowWriter

def _comma_fmt(x, pos):
    """Вспомогательная функция: меняет точку на запятую для осей."""
    return f"{x:g}".replace('.', ',')

class Visualizer:
    """Рисует графики и анимацию, используя данные от Solver."""

    def __init__(self, x, barrier, use_comma=True):
        """
        Инициализация.
        Параметр use_comma: если True — рисует с запятыми (рус), если False — с точками (англ).
        """
        self.x = np.asarray(x)
        self.barrier = barrier
        self.use_comma = use_comma
        plt.rcParams['font.family'] = 'DejaVu Sans'

    def _apply_format(self, ax):
        """Применяет выбранный формат к осям."""
        if self.use_comma:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_comma_fmt))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(_comma_fmt))
        # Если use_comma=False, matplotlib по умолчанию оставит точки.

    def plot_slices(self, times, densities, out_dir="results", targets=None):
        """Строит 3 графика: до, во время и после барьера."""
        Path(out_dir).mkdir(exist_ok=True)
        if targets is None:
            targets = [(0.0, "before"), (10.0, "during"), (15.0, "after")]

        for t_target, name in targets:
            i = int(np.argmin(np.abs(times - t_target)))
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.plot(self.x, densities[i], label=f"Время t = {times[i]:.2f}")
            ax.axvspan(0, self.barrier.a, color="red", alpha=0.2, label="Барьер")

            ax.set_xlabel("Координата x (атомные единицы)")
            ax.set_ylabel("Плотность вероятности |ψ(x,t)|²")
            ax.set_title(f"Эволюция волнового пакета (момент: {name})")
            ax.legend()
            ax.grid(alpha=0.3)
            self._apply_format(ax)

            fig.tight_layout()
            fig.savefig(f"{out_dir}/plot_{name}.png", dpi=300)
            plt.close(fig)

    def animate(self, times, densities, out_path="results/animation.gif", fps=20):
        """Создает GIF-анимацию."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        ymax = float(densities.max()) * 1.15
        fig, ax = plt.subplots(figsize=(8, 5))

        line, = ax.plot([], [], lw=1.5)
        ax.axvspan(0, self.barrier.a, color="red", alpha=0.25, label="Барьер")
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(0, ymax)

        ax.set_xlabel("Координата x (атомные единицы)")
        ax.set_ylabel("Плотность вероятности |ψ(x,t)|²")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        self._apply_format(ax)

        def update(i):
            line.set_data(self.x, densities[i])
            ax.set_title(f"Эволюция волнового пакета (t = {times[i]:.2f})")
            return line,

        anim = FuncAnimation(fig, update, frames=len(times), blit=True)
        anim.save(out_path, writer=PillowWriter(fps=fps))
        plt.close(fig)

    def plot_T_E(self, E_list, T_num, T_an=None, out_path="results/T_E.png"):
        """Строит график T(E)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(E_list, T_num, "o-", label="Численный расчет")
        if T_an is not None:
            ax.plot(E_list, T_an, "--", label="Аналитическая теория")

        ax.set_xlabel("Энергия частицы E (атомные единицы)")
        ax.set_ylabel("Коэффициент прохождения (T)")
        ax.set_title("Зависимость коэффициента прохождения от энергии")
        ax.grid(alpha=0.3)
        ax.legend()
        self._apply_format(ax)

        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    def plot_T_a(self, a_list, T_num, out_path="results/T_a.png"):
        """Строит график T(a)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(a_list, T_num, "o-", color="C1")

        ax.set_xlabel("Ширина барьера a (атомные единицы)")
        ax.set_ylabel("Коэффициент прохождения (T)")
        ax.set_title("Зависимость коэффициента прохождения от ширины барьера")
        ax.grid(alpha=0.3)
        self._apply_format(ax)

        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    def plot_T_V0(self, V0_list, T_num, out_path="results/T_V0.png"):
        """Строит график T(V0)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(V0_list, T_num, "o-", color="C2")

        ax.set_xlabel("Высота барьера V0 (атомные единицы)")
        ax.set_ylabel("Коэффициент прохождения (T)")
        ax.set_title("Зависимость коэффициента прохождения от высоты барьера")
        ax.grid(alpha=0.3)
        self._apply_format(ax)

        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    def plot_ln_T_a(self, a_list, T_num, out_path="results/lnT_a.png"):
        """Строит график ln(T) от a."""
        T_num = np.asarray(T_num)
        mask = T_num > 1e-12
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(a_list[mask], np.log(T_num[mask]), "o-", color="C3")

        ax.set_xlabel("Ширина барьера a (атомные единицы)")
        ax.set_ylabel("Логарифм коэффициента прохождения ln(T)")
        ax.set_title("Экспоненциальная зависимость от ширины барьера")
        ax.grid(alpha=0.3)
        self._apply_format(ax)

        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)