# Тесты на классы. Запуск: pytest -q tests/
import os
import sys

# добавляем корень проекта в sys.path, чтобы импортировался src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.potential import PotentialBarrier
from src.wavepacket import GaussianWavePacket
from src.solver import Params, SchrodingerSolver


def test_potential_barrier_value():
    """V(x) должен быть V0 внутри [0, a] и 0 снаружи."""
    b = PotentialBarrier(V0=1.5, a=2.0)
    x = np.linspace(-5, 5, 1001)
    V = b.value(x)
    assert np.allclose(V[x <= 0], 0.0)
    assert np.allclose(V[(x > 0) & (x < 2.0)], 1.5)
    assert np.allclose(V[x >= 2.0], 0.0)


def test_wavepacket_norm():
    """Стартовый пакет должен быть нормирован на 1."""
    x = np.linspace(-30, 30, 2048)
    dx = x[1] - x[0]
    wp = GaussianWavePacket(x0=-10.0, sigma=2.0, k0=1.0)
    psi = wp.initial_state(x, dx=dx)
    n = np.sum(np.abs(psi) ** 2) * dx
    assert abs(n - 1.0) < 1e-10


def test_solver_norm_conservation():
    """Норма сохраняется при эволюции (унитарность схемы)."""
    p = Params(steps=200, N=512, L=40, sigma=2.0)
    s = SchrodingerSolver(p, save_every=1000)
    n0 = s.norm()
    for _ in range(200):
        s.step()
    n1 = s.norm()
    assert abs(n1 - n0) < 1e-8


def test_solver_balance():
    """T + R + P должно быть близко к 1 после прогона."""
    p = Params(steps=3000, sigma=3.0, x0=-12.0)
    s = SchrodingerSolver(p, save_every=3000)
    s.run(verbose=False)
    T, R, P = s.balance()
    assert abs(T + R + P - 1.0) < 0.01


def test_free_packet():
    """Без барьера центр масс пакета уезжает вправо."""
    p = Params(V0=0.0, steps=500, sigma=2.0, N=512)
    s = SchrodingerSolver(p, save_every=1000)

    def x_cm():
        prob = np.abs(s.psi) ** 2
        return float(np.sum(s.x * prob) * s.dx)

    cm0 = x_cm()
    s.run(verbose=False)
    cm1 = x_cm()

    # k0 = 1, значит групповая скорость = 1, за t = 5 пройдёт ~5
    assert cm1 > cm0 + 3.0


def test_analytic_vs_numeric():
    """Численный T должен попадать в аналитику в туннельном режиме."""
    # при V0=1.0, a=0.5, E=0.5 аналитика даёт T ≈ 0.787
    E, V0, a = 0.5, 1.0, 0.5
    p = Params(V0=V0, a=a, E=E, sigma=6.0, x0=-20.0,
               L=80.0, N=2048, dt=0.01, steps=6000)
    s = SchrodingerSolver(p, save_every=6000)
    s.run(verbose=False)
    T_num, _, _ = s.balance()

    kappa = np.sqrt(2 * (V0 - E))
    T_an = 1.0 / (1.0 + V0 ** 2 * np.sinh(kappa * a) ** 2
                  / (4 * E * (V0 - E)))

    # допуск широкий: пакет имеет разброс по энергии, идеального совпадения
    # с моноэнергетической формулой не будет
    assert abs(T_num - T_an) < 0.3, f"T_num={T_num:.4f}, T_an={T_an:.4f}"