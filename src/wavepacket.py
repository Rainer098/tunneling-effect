"""Гауссов волновой пакет."""
import numpy as np


class GaussianWavePacket:
    """
    Гауссов пакет с центром x0, шириной sigma и средним импульсом k0.

        ψ(x,0) = (2πσ²)^(-1/4) · exp(-(x-x0)²/(4σ²) + i k0 x)
    """

    def __init__(self, x0, sigma, k0):
        assert sigma > 0, "sigma должна быть > 0"
        self.x0 = float(x0)
        self.sigma = float(sigma)
        self.k0 = float(k0)

    def initial_state(self, x, dx=None):
        """
        Считает ψ(x,0) на сетке x.

        Если dx передать — дополнительно нормирует по сетке. Это нужно,
        потому что гаусс на конечном отрезке обрезается, и дискретная
        норма уже не единица.
        """
        x = np.asarray(x)
        psi = (1.0 / (2 * np.pi * self.sigma ** 2)) ** 0.25 \
              * np.exp(-(x - self.x0) ** 2 / (4 * self.sigma ** 2)
                       + 1j * self.k0 * x)
        if dx is not None:
            psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        return psi