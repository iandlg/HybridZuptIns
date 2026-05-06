import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import Sequence

import src.online_correction.kalman_filter as kf


type ArrayLike = NDArray | Sequence[float | int]

#  inputs P_0, mu_0,
def sequential_fit(
        mu_0: NDArray,
        P_0: NDArray,
        Phi: NDArray,
        y: NDArray,
        sigma_n: float = 1.0,
):
    n_train, m = Phi.shape

    mu = np.zeros((n_train + 1, m))
    P  = np.zeros((n_train + 1, m, m))

    mu[0] = mu_0
    P[0]  = P_0

    R = np.array([[sigma_n**2]])     # (1, 1)

    for idx in range(n_train):
        H = Phi[idx][np.newaxis, :]  # (1, m)

        mu[idx + 1], P[idx + 1] = kf.measurement_update(
            state       = mu[idx],          # (m,)
            stateCov    = P[idx],           # (m, m)
            measurement = y[idx:idx+1],     # (1,)
            H           = H,                # (1, m)
            R           = R,                # (1, 1)
        )

    return mu, P
        