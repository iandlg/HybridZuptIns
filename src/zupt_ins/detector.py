import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from src.zupt_ins.initialization import INSConfig

def detector(u: NDArray, simdata: INSConfig) -> Tuple[NDArray, NDArray]:
    """
    Wrapper function for running the zero-velocity detection algorithm.

    Parameters
    ----------
    u : array_like, shape (6, N)
        IMU data matrix; rows are [ax, ay, az, wx, wy, wz], columns are samples.
    window_size : int
        Size of the sliding detection window.
    gamma : float
        Detection threshold.
    g : float
        Gravitational acceleration (m/s^2).
    sigma_a : float
        Accelerometer noise standard deviation.
    sigma_g : float
        Gyroscope noise standard deviation.

    Returns
    -------
    zupt : ndarray, shape (N,)
        Boolean vector of zero-velocity decisions (True = zero velocity).
    logL : ndarray, shape (N,)
        Log-likelihood of the test statistics.
    """
    u = np.asarray(u, dtype=float)
    N = u.shape[1]
    W = simdata.window_size

    zupt = np.zeros(N, dtype=bool)
    T = _glrt(u, W, simdata.g, simdata.sigma_a, simdata.sigma_g)

    # Mark windows where test statistic is below threshold as zero velocity
    for k in range(len(T)):
        if T[k] < simdata.gamma:
            zupt[k:k + W] = True

    # Pad edges before computing log-likelihood
    T_padded = np.concatenate([
        np.ones(int(W // 2), dtype=float) * T.max(),
        T,
        np.ones(int(W // 2), dtype=float) * T.max()
    ])

    logL = -W / 2.0 * T_padded

    return zupt, logL


def _glrt(
        u: NDArray,
        window_size: float,
        g: float,
        sigma_a: float,
        sigma_g: float
    ) -> NDArray:
    """
    Generalized likelihood ratio test (SHOE detector).

    Parameters
    ----------
    u : ndarray, shape (6, N)
        IMU data matrix; rows are [ax, ay, az, wx, wy, wz], columns are samples.
    window_size : int
        Size of the sliding detection window.
    g : float
        Gravitational acceleration (m/s^2).
    sigma_a : float
        Accelerometer noise standard deviation.
    sigma_g : float
        Gyroscope noise standard deviation.

    Returns
    -------
    T : ndarray, shape (N - W + 1,)
        Test statistics for each window position.
    """
    W        = window_size
    N        = u.shape[1]
    sigma2_a = sigma_a**2
    sigma2_g = sigma_g**2

    T = np.zeros(N - W + 1, dtype=float)

    for k in range(N - W + 1):
        window_acc = u[0:3, k:k + W]           # (3, W)
        window_gyr = u[3:6, k:k + W]           # (3, W)

        ya_m = window_acc.mean(axis=1)          # (3,)
        g_ref = g * ya_m / np.linalg.norm(ya_m)

        acc_residuals = window_acc - g_ref[:, np.newaxis]   # (3, W)

        T[k] = (
            np.sum(window_gyr ** 2) / sigma2_g +
            np.sum(acc_residuals ** 2) / sigma2_a
        ) / W
    return T


if __name__ == "__main__": 
    from src.zupt_ins.data_classes import InertialData
    from src.zupt_ins.initialization import INSConfig

    cfg = INSConfig()
    imu_data = InertialData.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)

    zupt, logL = detector(imu_data.u, cfg)

    idx = 988
    print(f"{zupt[idx] = }")
    print(f"{logL[idx] = }")

    import matplotlib.pyplot as plt

    plt.plot(zupt*np.abs(logL).max())
    plt.plot(logL)
    plt.show()

