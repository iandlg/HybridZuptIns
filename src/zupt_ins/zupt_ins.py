
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from src.orientation import orientation

def comp_internal_states(x_in: NDArray, dx: NDArray, q_in: NDArray)-> Tuple[NDArray, NDArray]:
    """
    Correct navigation states with Kalman filter estimated perturbations.

    Parameters
    ----------
    x_in  : (9, N)  A priori navigation state vectors
    dx    : (9, N)  System perturbation vectors from Kalman filter
    q_in  : (4, N)  A priori quaternion vectors [qx, qy, qz, qw]

    Returns
    -------
    x_out : (9, N)  Corrected (posteriori) navigation state vectors
    q_out : (4, N)  Corrected quaternion vectors
    """


    N = x_in.shape[1]

    # --- Convert quaternion to rotation matrix (3, 3, N) ---
    R = orientation.q2dcm(q_in)

    # --- Correct the full state vector ---
    x_out = x_in + dx                           # (9, N)

    # --- Build skew-symmetric OMEGA for each sample ---
    epsilon = dx[6:9, :]                        # (3, N)
    
    # OMEGA[i] = [[0, -e3, e2], [e3, 0, -e1], [-e2, e1, 0]]
    zeros = np.zeros(N)
    OMEGA = np.array([
        [ zeros,       -epsilon[2],  epsilon[1]],
        [ epsilon[2],   zeros,       -epsilon[0]],
        [-epsilon[1],   epsilon[0],   zeros     ]
    ])                                          # (3, 3, N)

    # --- Correct rotation matrix: R = (I - OMEGA) @ R ---
    I = np.eye(3)[:, :, np.newaxis]             # (3, 3, 1) broadcasts over N
    R = np.einsum('ijk,jlk->ilk', I - OMEGA, R) # (3, 3, N)

    # --- Extract Euler angles from corrected rotation matrix ---
    x_out[6] = np.arctan2(R[2, 1], R[2, 2])                              # roll
    x_out[7] = np.arctan(-R[2, 0] / np.sqrt(R[2, 1]**2 + R[2, 2]**2))   # pitch
    x_out[8] = np.arctan2(R[1, 0], R[0, 0])                              # heading

    # --- Recover corrected quaternions ---
    q_out = orientation.dcm2q(R)                            # (4, N)

    return x_out, q_out