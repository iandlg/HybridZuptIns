from numpy.typing import NDArray
import numpy as np

def rotation_stack_to_euler(R: NDArray) -> NDArray:
    """
    Convert a stack of rotation matrices to Euler angles using the convention:

        r = atan2(R32, R33)
        p = -asin(R31)
        y = atan2(R21, R11)

    Args:
        R : NDArray (3, 3, N)

    Returns:
        NDArray (3, N)  -> [roll, pitch, yaw]
    """

    r = np.arctan2(R[2, 1, :], R[2, 2, :])
    p = -np.arcsin(R[2, 0, :])
    y = np.arctan2(R[1, 0, :], R[0, 0, :])

    return np.vstack((r, p, y))

def q2dcm(q: NDArray):
    """
    Convert quaternion(s) to Direction Cosine Matrix (DCM).
    
    Parameters:
        q: array of shape (4,) or (4, N) — quaternions [q1, q2, q3, q4]
    
    Returns:
        R: array of shape (3, 3) or (3, 3, N) — rotation matrices
    """
    q = np.atleast_2d(q)
    if q.shape[0] == 1 and q.shape[1] == 4:
        q = q.T  # handle (1,4) input
    
    single = q.shape[1] == 1 if q.ndim == 2 else False
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

    return R[:, :, 0] if single else R