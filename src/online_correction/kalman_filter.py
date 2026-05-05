import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def dynamic_update(
        states: NDArray,
        stateCovs: NDArray,
        F: NDArray, Q: NDArray
    ):
    """
    Vectorized Kalman prediction step for multiple states.

    states:    (N, dim)
    stateCovs: (N, dim, dim)
    F:         (dim, dim)
    Q:         (dim, dim)
    """
    # Predict state: for each state row i → F @ states[i]
    updated_states = states @ F.T                 # (N, dim)

    # Predict covariance: F P Fᵀ + Q for each state
    updated_covs = F @ stateCovs @ F.T + Q        # broadcasting → (N, dim, dim)

    return updated_states, updated_covs

def measurement_update(
        state: NDArray,
        stateCov: NDArray,
        measurement: NDArray,
        H: NDArray,
        R: NDArray
    ) -> Tuple[NDArray, NDArray]:
    """
    Kalman measurement update for a single state.

    Parameters
    ----------
    state      : NDArray, shape (dim_x,)
    stateCov   : NDArray, shape (dim_x, dim_x)
    measurement: NDArray, shape (dim_z,)
    H          : NDArray, shape (dim_z, dim_x)
    R          : NDArray, shape (dim_z, dim_z)

    Returns
    -------
    updated_state : NDArray, shape (dim_x,)
    updated_cov   : NDArray, shape (dim_x, dim_x)
    """
    # Innovation
    innovation = measurement - H @ state             # (dim_z,)

    # Innovation covariance
    S = H @ stateCov @ H.T + R                       # (dim_z, dim_z)

    # Kalman gain
    K = stateCov @ H.T @ np.linalg.inv(S)            # (dim_x, dim_z)

    # Updated state
    updated_state = state + K @ innovation            # (dim_x,)

    # Updated covariance
    I = np.eye(state.shape[0])
    updated_cov = (I - K @ H) @ stateCov             # (dim_x, dim_x)

    return updated_state, updated_cov