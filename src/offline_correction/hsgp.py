import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Sequence

import src.zupt_ins.orientation as orientation 
from src.zupt_ins.data_classes import Trajectory, TimeSeries
from sklearn.preprocessing import StandardScaler

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
    
def power_spectral_density(
        omega: NDArray,
        n_dims: int,
        ls: float | NDArray[np.floating],
        sigma_f: float = 1.0
    ) -> NDArray:
    """
    Power spectral density for the ExpQuad kernel.

    .. math::

        S(\\boldsymbol\\omega) =
            (\\sqrt(2 \\pi)^D \\prod_{i}^{D}\\ell_i
            \\exp\\left( -\\frac{1}{2} \\sum_{i}^{D}\\ell_i^2 \\omega_i^{2} \\right)

    Args:
        omega: array of shape (m_star, d), frequencies at which to evaluate the PSD.
        ls:    lengthscale(s), either a scalar or array of shape (d,).
        n_dims: number of input dimensions D.

    Returns:
        Array of shape (m_star,), one PSD value per basis function.
    """
    ls_arr = np.ones(n_dims) * ls          # (d,)
    c = np.power(np.sqrt(2.0 * np.pi), n_dims)
    exp = np.exp(-0.5 * np.dot(np.square(omega), np.square(ls_arr)))  # (m_star,)
    return sigma_f**2 * c * np.prod(ls_arr) * exp       # (m_star,)

def calc_eigenvalues(L: NDArray, m: Sequence[int]) -> NDArray:
    """Calculate eigenvalues of the Laplacian."""
    temp = [np.arange(1, 1 + m[d]) for d in range(len(m))]
    S = np.meshgrid(*temp)
    S_arr = np.vstack([s.flatten() for s in S]).T
    return np.square((np.pi * S_arr) / (2 * L))

def calc_eigenvectors(Xs: NDArray, L: NDArray, eigvals: NDArray) -> NDArray:
    """Calculate eigenvectors of the Laplacian.
    These are used as basis vectors in the HSGP approximation.

    Parameters
    ----------
    Xs      : NDArray, shape (n_samples, d)
    L       : NDArray, shape (d,)
    eigvals : NDArray, shape (m_star, d)
    m       : Sequence[int], shape (d,)

    Returns
    -------
    phi : NDArray, shape (n_samples, m_star)
    """
    # (1, m_star, d) * (n_samples, 1, d) -> (n_samples, m_star, d)
    term1 = np.sqrt(eigvals)[None, :, :]          # (1, m_star, d)
    term2 = Xs[:, None, :] + L[None, None, :]     # (n_samples, 1, d) + (1, 1, d)
    c = 1.0 / np.sqrt(L)                          # (d,)

    phi = c * np.sin(term1 * term2)               # (n_samples, m_star, d)
    return np.prod(phi, axis=-1)                  # (n_samples, m_star)


def compute_hsgp_corrections(
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        m: Sequence[int] | int = 25,
        ls: float | NDArray[np.floating] = 1.0,
        sigma_f: float = 1.0,
        sigma_n: float = 1.0,
        L: float | NDArray[np.floating] = 1.5
    ) -> NDArray:
    """
    Estimate output corrections using leave-one-fold-out Gaussian Process regression.

    Performs 10-fold cross-validation, fitting a GP with an RBF + white noise
    kernel on each fold. Also computes a static (mean) correction as a baseline.

    Parameters
    -------
    x : NDArray[np.floating], shape (n_features, n_samples)
        Input features, one column per sample.
    y : NDArray[np.floating], shape (n_samples,)
        Scalar target values to regress.

    Returns
    -------
    y_testing_GP : NDArray[np.floating], shape (n_samples,)
        GP-predicted corrections assembled across all folds.
    y_testing_static : NDArray[np.floating], shape (n_samples,)
        Static correction, equal to the global mean of y for all samples.
    """
    n_samples = x.shape[1]
    n_dim = x.shape[0]
    y_testing_GP = np.zeros(n_samples)
    D = 10

    if isinstance(m, int) : 
        m = [m]*n_dim
    
    if isinstance(L, float):
        L = np.ones(n_dim, dtype=float) * L

    eigvals = calc_eigenvalues(L, m)  # type: ignore # (m_start, d)

    # Scale inputs to unit variance — critical for RBF length scale to be meaningful
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.T)  # (n_samples, n_features)

    for i in range(1, D + 1):
        # --- indices --------------------------------------------------------
        test_start = int(np.floor(n_samples / D * (i - 1)))
        test_end   = int(np.floor(n_samples / D * i))

        training_ind = list(range(0, test_start)) + list(range(test_end, n_samples))
        testing_ind  = list(range(test_start, test_end))

        # --- data -----------------------------------------------------------
        x_train = x_scaled[training_ind,:]   # (n_train, n_features)
        y_train = y[training_ind]        # (n_train, )
        x_test  = x_scaled[testing_ind, :]    # (n_test,  n_features)

        # --- fit + predict --------------------------------------------------
        phi = calc_eigenvectors(x_train, L, eigvals)  # (n_samples, m_star)
        phi_star = calc_eigenvectors(x_test, L, eigvals)
        omega = np.sqrt(eigvals)  # (m_star, d)
        psd = power_spectral_density(omega, n_dim, ls, sigma_f) 

        A = phi.T @ phi + sigma_n**2 * np.diag(1 / psd)
        L_chol = np.linalg.cholesky(A)                          # (m_star, m_star)
        alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, phi.T @ y_train))

        y_testing_GP[testing_ind] = phi_star @ alpha                             # predictive mean
        # var = sigma_f**2 - np.sum(phi_star @ np.linalg.solve(L_chol, phi_star.T).T, axis=1)  # predictive variance                

    return y_testing_GP


if __name__ == "__main__" : 
    pass