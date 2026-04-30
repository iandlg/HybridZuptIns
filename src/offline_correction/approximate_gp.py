import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional

import src.zupt_ins.orientation as orientation 
from src.zupt_ins.data_classes import Trajectory, TimeSeries

class SquaredExponentialKernel:
    def __init__(self, sigma_f: float = 1.0, len_scale: float = 1.0):
        self.sigma_f = sigma_f
        self.len_scale = len_scale

    def __call__(self, A: NDArray, B: NDArray) -> NDArray:
        A = np.asarray(A)  # (n_a, n_features)
        B = np.asarray(B)  # (n_b, n_features)

        # ||xa - xb||^2, shape (n_a, n_b)
        sqdist = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)

        return self.sigma_f**2 * np.exp(-sqdist / (2 * self.len_scale**2))

    def spectral_density(self, s: NDArray, d: int) -> NDArray:
        """
        Spectral density of the SE kernel (eq. 3-30):
            S_SE(s) = sigma_f^2 * (2*pi*l^2)^(d/2) * exp(-2*pi^2*l^2*s^2)

        Parameters
        ----------
        s : NDArray
            Frequencies at which to evaluate the spectral density.
        d : int
            Dimension of the input x.
        """
        s = np.asarray(s)
        return (
            self.sigma_f**2
            * (2 * np.pi * self.len_scale**2) ** (d / 2)
            * np.exp(-2 * np.pi**2 * self.len_scale**2 * s**2)
        )