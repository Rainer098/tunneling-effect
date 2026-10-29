"""Графики и анимация. Ничего не считает — только рисует то, что дали."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.potential import PotentialBarrier

FloatArray = NDArray[np.float64]


class Visualizer:
    """
    Рисовалка. Хранит сетку x и барьер, чтобы каждый раз не таскать их
    параметрами в методы.
    """

    def __init__(self, x: FloatArray, barrier: PotentialBarrier) -> None:
        self.x: FloatArray = np.asarray(x)
        self.barrier: PotentialBarrier = barrier

    # ---------- одиночные картинки ----------

    def plot_wavefunction(
        self,
        prob: FloatArray,
        title: str = "",
        out_path: str | None = None,
    ) -> None:
        """Строит |ψ(x)|² для одного момента времени."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.x, prob, label=r"$|\psi|^2$")
        plt.axvspan(0, self.barrier.a, color="red", alpha=0.2, label="барьер")
        plt.xlabel("x")
        plt.ylabel(r"$|\psi(x)|^2$")
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_slices(
        self,
        times: FloatArray,
        densities: FloatArray,
        out_dir: str = "results",
        targets: list[tuple[float, str]] | None = None,
    ) -> None:
        """
        Режет эволюцию по трём опорным моментам: до удара, в момент удара,
        после. По умолчанию t = 0, 10, 15.
        """
        Path(out_dir).mkdir(exist_ok=True)
        if targets is None:
            targets = [(0.0, "before"), (10.0, "during"), (15.0, "after")]

        for t_target, name in targets:
            i: int = int(np.argmin(np.abs(times - t_target)))

            plt.figure(figsize=(8, 5))
            plt.plot(self.x, densities[i], label=f"t = {times[i]:.2f}")
            plt.axvspan(0, self.barrier.a, color="red", alpha=0.2,
                        label="барьер")
            plt.xlabel("x")
            plt.ylabel(r"$|\psi(x,t)|^2$")
            plt.title(name)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{out_dir}/plot_{name}.png", dpi=300)
            plt.close()

    # ---------- анимация ----------

    def animate(
        self,
        times: FloatArray,
        densities: FloatArray,
        out_path: str = "results/animation.gif",
        fps: int = 20,
    ) -> None:
        """
        Собирает GIF: |ψ(x,t)|² по кадрам. Барьер рисуется красной полосой
        поверх каждого кадра.
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        ymax: float = float(densities.max()) * 1.15
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot([], [], lw=1.5)
        ax.axvspan(0, self.barrier.a, color="red", alpha=0.25,
                   label="барьер")
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(0, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$|\psi(x,t)|^2$")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

        def update(i: int):
            line.set_data(self.x, densities[i])
            ax.set_title(f"t = {times[i]:.2f}")
            return line,

        anim = FuncAnimation(fig, update, frames=len(times), blit=True)
        anim.save(out_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
        print(f"анимация сохранена: {out_path}")

    # ---------- зависимости T от параметров ----------

    def plot_T_E(
        self,
        E_list: FloatArray,
        T_num: FloatArray,
        T_an: FloatArray | None = None,
        out_path: str = "results/T_E.png",
    ) -> None:
        """T(E). Если дана аналитика — рисуем её пунктиром."""
        plt.figure(figsize=(8, 5))
        plt.plot(E_list, T_num, "o-", label="численно")
        if T_an is not None:
            plt.plot(E_list, T_an, "--", label="аналитика")
        plt.xlabel("E")
        plt.ylabel("T")
        plt.title("Коэффициент прохождения от энергии")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_T_a(
        self,
        a_list: FloatArray,
        T_num: FloatArray,
        out_path: str = "results/T_a.png",
    ) -> None:
        """T(a) — при фиксированных E и V0."""
        plt.figure(figsize=(8, 5))
        plt.plot(a_list, T_num, "o-", color="C1")
        plt.xlabel("a")
        plt.ylabel("T")
        plt.title("Коэффициент прохождения от ширины барьера")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_T_V0(
        self,
        V0_list: FloatArray,
        T_num: FloatArray,
        out_path: str = "results/T_V0.png",
    ) -> None:
        """T(V0) — при фиксированных E и a."""
        plt.figure(figsize=(8, 5))
        plt.plot(V0_list, T_num, "o-", color="C2")
        plt.xlabel("V0")
        plt.ylabel("T")
        plt.title("Коэффициент прохождения от высоты барьера")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_ln_T_a(
        self,
        a_list: FloatArray,
        T_num: FloatArray,
        out_path: str = "results/lnT_a.png",
    ) -> None:
        """ln(T) от a — наклон прямой должен быть −2κ."""
        T_arr: FloatArray = np.asarray(T_num)
        # крошечные T (< 1e-12) в лог не лезут, отсекаем
        mask: NDArray = T_arr > 1e-12
        plt.figure(figsize=(8, 5))
        plt.plot(a_list[mask], np.log(T_arr[mask]), "o-", color="C3")
        plt.xlabel("a")
        plt.ylabel(r"$\ln T$")
        plt.title(r"$\ln T$ от ширины барьера")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
        plt.close()