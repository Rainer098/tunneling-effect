"""Split-operator для 1D нестационарного уравнения Шрёдингера."""
from dataclasses import dataclass, asdict

import numpy as np

from src.potential import PotentialBarrier
from src.wavepacket import GaussianWavePacket
from src.data_manager import DataManager


@dataclass
class Params:
    """Параметры задачи. Значения по умолчанию — как в исходнике."""
    V0:    float = 1.0
    a:     float = 0.5
    E:     float = 0.5
    sigma: float = 1.0
    x0:    float = -10.0
    L:     float = 40.0
    N:     int   = 1024
    dt:    float = 0.01
    steps: int   = 2000


class SchrodingerSolver:
    """
    Эволюция ψ по split-operator'у:
        ψ(t+dt) = exp(-iV·dt/2) · F⁻¹[exp(-ik²·dt/2) · F[ψ]] · exp(-iV·dt/2)

    Кинетика делается в k-пространстве через FFT.
    """

    def __init__(self, params: Params, save_every: int = 10) -> None:
        self.p = params
        self.save_every = save_every

        # сетка по x и сопряжённые частоты
        self.x = np.linspace(-params.L / 2, params.L / 2, params.N, endpoint=False)
        self.dx = params.L / params.N
        # fftfreq даёт 1/длину, домножаем на 2π, чтобы получить k
        self.k = 2 * np.pi * np.fft.fftfreq(params.N, d=self.dx)

        # барьер и стартовый пакет
        self.barrier = PotentialBarrier(params.V0, params.a)
        self.V = self.barrier.value(self.x)

        k0 = np.sqrt(2 * params.E)
        packet = GaussianWavePacket(params.x0, params.sigma, k0)
        self.psi = packet.initial_state(self.x, dx=self.dx)

        # сюда пишем снимки для анимации/графиков
        self.times = []
        self.densities = []

    # ---------- базовые операции ----------

    def norm(self) -> float:
        """Текущая норма ∫|ψ|² dx."""
        return float(np.sum(np.abs(self.psi) ** 2) * self.dx)

    def check_norm(self, tol: float = 1e-8) -> float:
        """Проверяет, что норма не уехала (инвариант унитарной эволюции)."""
        n = self.norm()
        assert abs(n - 1.0) < tol, f"норма уехала: {n}"
        return n

    def step(self) -> None:
        """Один шаг по времени на dt."""
        dt = self.p.dt
        # полшага потенциала
        self.psi = np.exp(-1j * self.V * dt / 2) * self.psi
        # кинетика: прыгаем в k-пространство, там exp(-ik²dt/2), обратно
        self.psi = np.fft.ifft(
            np.exp(-1j * self.k ** 2 * dt / 2) * np.fft.fft(self.psi)
        )
        # ещё полшага потенциала
        self.psi = np.exp(-1j * self.V * dt / 2) * self.psi

    def run(self, verbose: bool = True) -> None:
        """Крутит цикл по времени, попутно сохраняя |ψ|² каждые save_every шагов."""
        self.check_norm()

        for n in range(self.p.steps + 1):
            if n % self.save_every == 0:
                self.times.append(n * self.p.dt)
                self.densities.append(np.abs(self.psi) ** 2)

            if n == self.p.steps:
                break

            self.step()
            self.check_norm()

            if verbose and n % 500 == 0:
                print(f"шаг {n}/{self.p.steps},  t = {n * self.p.dt:.2f}")

        self.times = np.array(self.times)
        self.densities = np.array(self.densities)

    # ---------- физика ----------

    def balance(self) -> tuple[float, float, float]:
        """
        T — доля вероятности справа от барьера,
        R — слева,
        P — внутри барьера.
        """
        prob = np.abs(self.psi) ** 2
        x, dx, a = self.x, self.dx, self.p.a
        T = float(np.sum(prob[x > a]) * dx)
        R = float(np.sum(prob[x < 0]) * dx)
        P = float(np.sum(prob[(x >= 0) & (x <= a)]) * dx)
        return T, R, P

    # ---------- сохранение ----------

    def save(self, out_dir: str = "results") -> None:
        """Скидывает параметры и плотности на диск через DataManager."""
        dm = DataManager(out_dir)
        dm.save_params(asdict(self.p))
        dm.save_data(self.densities, "densities.npy")