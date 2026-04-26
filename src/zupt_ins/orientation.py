from numpy.typing import NDArray
import numpy as np

def matrix_to_euler(R: NDArray) -> NDArray:
    """
    Convert a stack of rotation matrices to Euler angles using the convention:

        r = atan2(R32, R33)
        p = -asin(R31)
        y = atan2(R21, R11)

    Args:
        R : NDArray (3, 3, N) or (3, 3)
            Rotation matrices. If (3, 3), returns (3,) Euler angles.

    Returns:
        NDArray (3, N) or (3,)  -> [roll, pitch, yaw]
    """
    R = np.asarray(R, dtype=float)
    single = R.ndim == 2
    if single:
        R = R[:, :, np.newaxis]  # (3, 3, 1)

    r = np.arctan2(R[2, 1, :], R[2, 2, :])
    p = -np.arcsin(R[2, 0, :])
    y = np.arctan2(R[1, 0, :], R[0, 0, :])

    result = np.vstack((r, p, y))
    return result[:, 0] if single else result

def euler_to_matrix(ang: NDArray[np.floating])->NDArray[np.floating]:
    """
    Compute rotation matrices from Euler angles.

    Parameters
    ----------
    ang : array_like, shape (3, N) or (3,)
        Euler angles [roll, pitch, yaw] in radians.
        If shape (3,), a single rotation matrix is returned.

    Returns
    -------
    R : ndarray, shape (3, 3, N) or (3, 3)
        Rotation matrix/matrices corresponding to the input Euler angles.
    """
    ang = np.asarray(ang, dtype=float)
    single = ang.ndim == 1
    if single:
        ang = ang[:, np.newaxis]  # (3, 1)

    cr, sr = np.cos(ang[0]), np.sin(ang[0])
    cp, sp = np.cos(ang[1]), np.sin(ang[1])
    cy, sy = np.cos(ang[2]), np.sin(ang[2])

    # Build (3, 3, N) array — each [:, :, k] is one rotation matrix (transposed)
    N = ang.shape[1]
    R = np.empty((3, 3, N))

    R[0, 0] =  cy*cp
    R[0, 1] =  sy*cp
    R[0, 2] = -sp
    R[1, 0] = -sy*cr + cy*sp*sr
    R[1, 1] =  cy*cr + sy*sp*sr
    R[1, 2] =  cp*sr
    R[2, 0] =  sy*sr + cy*sp*cr
    R[2, 1] = -cy*sr + sy*sp*cr
    R[2, 2] =  cp*cr

    # Transpose each matrix: swap axes 0 and 1
    R = R.transpose(1, 0, 2)

    return R[:, :, 0] if single else R

def q2dcm(q: NDArray):
    """
    Convert quaternion(s) to Direction Cosine Matrix (DCM).
    
    Parameters:
        q: array of shape (4, N) — quaternions [q1, q2, q3, q4]
    
    Returns:
        R: array of shape (3, 3) or (3, 3, N) — rotation matrices
    """
    q = np.asarray(q, dtype=float)
    single = q.ndim == 1
    if single:
        q = q[:, np.newaxis]  # (4, 1)
    
    N = q.shape[1]

    q1, q2, q3, q4 = q[0], q[1], q[2], q[3]

    p1 = q1**2
    p2 = q2**2
    p3 = q3**2
    p4 = q4**2
    p5 = p2 + p3
    denom = p1 + p4 + p5

    p6 = np.where(denom != 0, 2.0 / denom, 0.0)

    R = np.zeros((3, 3, N))

    R[0, 0] = 1 - p6 * p5
    R[1, 1] = 1 - p6 * (p1 + p3)
    R[2, 2] = 1 - p6 * (p1 + p2)

    a1 = p6 * q1
    a2 = p6 * q2

    t = p6 * q3 * q4
    u = a1 * q2
    R[0, 1] = u - t
    R[1, 0] = u + t

    t = a2 * q4
    u = a1 * q3
    R[0, 2] = u + t
    R[2, 0] = u - t

    t = a1 * q4
    u = a2 * q3
    R[1, 2] = u - t
    R[2, 1] = u + t

    return R[:,:,0] if single else R

def dcm2q(R: NDArray)->NDArray:
    """
    Convert directional cosine matrix (rotation matrix) to quaternion vector.

    Args:
        R : np.ndarray
            Rotation matrix, shape (3, 3, N) or (3, 3).
            If (3, 3), returns (4,) quaternion.

    Returns:
        q : np.ndarray
            Quaternion vector [qx, qy, qz, qw], shape (4,) or (4, N)
    """
    R = np.asarray(R, dtype=float)
    single = R.ndim == 2
    if single:
        R = R[:, :, np.newaxis]  # (3, 3, 1)

    N = R.shape[2]
    q = np.zeros((4, N))

    T = 1 + R[0, 0] + R[1, 1] + R[2, 2]  # (N,)

    # --- Case 1: T > 1e-8 ---
    mask1 = T > 1e-8
    if np.any(mask1):
        S = 0.5 / np.sqrt(T[mask1])
        q[3, mask1] = 0.25 / S
        q[0, mask1] = (R[2, 1, mask1] - R[1, 2, mask1]) * S
        q[1, mask1] = (R[0, 2, mask1] - R[2, 0, mask1]) * S
        q[2, mask1] = (R[1, 0, mask1] - R[0, 1, mask1]) * S

    # --- Case 2: T <= 1e-8 ---
    mask2 = ~mask1

    # Sub-case 2a: R[0,0] is dominant diagonal
    mask2a = mask2 & (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2])
    if np.any(mask2a):
        S = np.sqrt(1 + R[0, 0, mask2a] - R[1, 1, mask2a] - R[2, 2, mask2a]) * 2  # S = 4*qx
        q[3, mask2a] = (R[2, 1, mask2a] - R[1, 2, mask2a]) / S
        q[0, mask2a] = 0.25 * S
        q[1, mask2a] = (R[0, 1, mask2a] + R[1, 0, mask2a]) / S
        q[2, mask2a] = (R[0, 2, mask2a] + R[2, 0, mask2a]) / S

    # Sub-case 2b: R[1,1] is dominant diagonal
    mask2b = mask2 & ~mask2a & (R[1, 1] > R[2, 2])
    if np.any(mask2b):
        S = np.sqrt(1 + R[1, 1, mask2b] - R[0, 0, mask2b] - R[2, 2, mask2b]) * 2  # S = 4*qy
        q[3, mask2b] = (R[0, 2, mask2b] - R[2, 0, mask2b]) / S
        q[0, mask2b] = (R[0, 1, mask2b] + R[1, 0, mask2b]) / S
        q[1, mask2b] = 0.25 * S
        q[2, mask2b] = (R[1, 2, mask2b] + R[2, 1, mask2b]) / S

    # Sub-case 2c: R[2,2] is dominant diagonal
    mask2c = mask2 & ~mask2a & ~mask2b
    if np.any(mask2c):
        S = np.sqrt(1 + R[2, 2, mask2c] - R[0, 0, mask2c] - R[1, 1, mask2c]) * 2  # S = 4*qz
        q[3, mask2c] = (R[1, 0, mask2c] - R[0, 1, mask2c]) / S
        q[0, mask2c] = (R[0, 2, mask2c] + R[2, 0, mask2c]) / S
        q[1, mask2c] = (R[1, 2, mask2c] + R[2, 1, mask2c]) / S
        q[2, mask2c] = 0.25 * S

    return q[:, 0] if single else q

