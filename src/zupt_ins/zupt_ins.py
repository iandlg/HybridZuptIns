import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Any
import scipy.linalg as  linalg
from math import factorial

from src.zupt_ins import orientation
from src.zupt_ins import detector
from src.zupt_ins.data_classes import InertialData, Trajectory
from src.zupt_ins.initialization import INSConfig

class StepDetector:
    def __init__(self):
        self.zupt_counter = 0
        self.no_zupt_counter = 0
        self.reached_no_zupt_counter = False

    def __call__(self, k: int, z: int) -> Optional[int]:
        if z == 1:
            self.zupt_counter += 1
            self.no_zupt_counter = 0
        else:
            self.no_zupt_counter += 1
            self.zupt_counter = 0

        if self.zupt_counter == 10 and self.reached_no_zupt_counter:
            self.reached_no_zupt_counter = False
            return k - 2

        if self.no_zupt_counter == 50:
            self.reached_no_zupt_counter = True

        return None

def smoothed_zupt_aided_ins(
        inertial: InertialData,
        simdata: INSConfig,
        online_corrector: Any = None
    ) -> Tuple[NDArray, Trajectory, List[int]]:
    """
    Run the open-loop zero-velocity aided INS Kalman filter with RTS smoothing.

    Parameters
    ----------
    inertial : dict
        Dictionary containing:
            - 'u'    : ndarray, shape (6, N), IMU data matrix.
            - 'zupt' : ndarray, shape (N,), zero-velocity decisions (bool).
    simdata : object or dict
        Settings containing:
            - sigma_initial_pos : array_like, shape (3,)
            - sigma_initial_vel : array_like, shape (3,)
            - sigma_initial_att : array_like, shape (3,)
            - sigma_acc         : array_like, shape (3,)
            - sigma_gyro        : array_like, shape (3,)
            - sigma_vel         : array_like, shape (3,)
            - init_heading      : float
            - init_pos          : array_like, shape (3,)
    Ts : float
        Sampling time (seconds).
    g : float
        Gravitational acceleration (m/s^2).

    Returns
    -------
    inertial : dict
        Updated dictionary with added fields:
            - 'pos' : ndarray, shape (3, N), estimated positions.
            - 'R'   : ndarray, shape (3, 3, N), rotation matrices per timestep.
    """
    u = inertial.u
    Ts = simdata.Ts
    g = simdata.g

    zupt, _ = detector.detector(u, simdata)

    N = len(zupt)

    # Initialise filter matrices
    Q, R, H = init_filter(simdata)
    Id = np.eye(9)

    # Allocate state arrays
    x            = np.zeros((9, N))
    quat         = np.zeros((4, N))
    dx           = np.zeros((9, N))
    dx_timeupd   = np.zeros((9, N))
    dx_smooth    = np.zeros((9, N))

    # Allocate covariance arrays
    cov          = np.zeros((9, N))
    cov_smooth   = np.zeros((9, N))
    P            = np.zeros((9, 9, N))
    P_timeupd    = np.zeros((9, 9, N))
    P_smooth     = np.zeros((9, 9, N))
    F            = np.zeros((9, 9, N))

    # Initialise covariance matrix
    P[0:3, 0:3, 0] = np.diag(simdata.sigma_initial_pos_array**2)
    P[3:6, 3:6, 0] = np.diag(simdata.sigma_initial_vel_array**2)
    P[6:9, 6:9, 0] = np.diag(simdata.sigma_initial_att_array**2)
    cov[:, 0]      = np.diag(P[:, :, 0])

    # Initialise navigation state
    x[:, 0], quat[:, 0] = initialize_nav(u, simdata.init_heading, simdata.init_pos_array)

    # Segment bookkeeping
    seg_start = 1
    seg_end   = N - 1
    step_detector = StepDetector()
    step_seg = []

    while True:

        # ------------------------------------------------------------------ #
        # Forward Kalman filter
        # ------------------------------------------------------------------ #

        for n in range(seg_start, seg_end + 1):

            # Time update -------------------------------------------------- #
            x[:, n], quat[:, n] = navigation_equations(
                x[:, n - 1], u[:, n], quat[:, n - 1], Ts, g
            )
            
            F[:, :, n], G = state_matrix(quat[:, n], u[:, n], Ts)

            dx[:, n]     = F[:, :, n] @ dx[:, n - 1]
            P[:, :, n]   = F[:, :, n] @ P[:, :, n - 1] @ F[:, :, n].T + G @ Q @ G.T
            
            dx_timeupd[:, n]   = dx[:, n]
            P_timeupd[:, :, n] = P[:, :, n]

            # Zero-velocity update ----------------------------------------- #
            if zupt[n]:
                K            = (P[:, :, n] @ H.T) @ np.linalg.inv(H @ P[:, :, n] @ H.T + R)
                dx[:, n]     = dx[:, n] - K @ (dx[3:6, n] - x[3:6, n])
                P[:, :, n]   = (Id - K @ H) @ P[:, :, n]

            # Symmetrise
            P[:, :, n] = (P[:, :, n] + P[:, :, n].T) / 2
            cov[:, n]  = np.diag(P[:, :, n])

            # # Segmentation decision ---------------------------------------- #
            detected = step_detector(n, zupt[n])
            if detected is not None:
                step_seg.append(detected)
                seg_end = n
                break
            

        # ------------------------------------------------------------------ #
        # RTS smoothing
        # ------------------------------------------------------------------ #
        dx_smooth[:, seg_end]    = dx[:, seg_end]
        P_smooth[:, :, seg_end]  = P[:, :, seg_end]
        cov_smooth[:, seg_end]   = np.diag(P_smooth[:, :, seg_end])

        for n in range(seg_end - 1, seg_start - 1, -1):
            A = P[:, :, n] @ F[:, :, n].T @ np.linalg.inv(P_timeupd[:, :, n + 1])

            dx_smooth[:, n]   = dx[:, n] + A @ (dx_smooth[:, n + 1] - dx_timeupd[:, n + 1])
            P_smooth[:, :, n] = P[:, :, n] + A @ (P_smooth[:, :, n + 1] - P_timeupd[:, :, n + 1]) @ A.T
            P_smooth[:, :, n] = (P_smooth[:, :, n] + P_smooth[:, :, n].T) / 2
            cov_smooth[:, n]  = np.diag(P_smooth[:, :, n])

        # ------------------------------------------------------------------ #
        # Internal state compensation
        # ------------------------------------------------------------------ #

        x[:,seg_start:seg_end+1], quat[:,seg_start:seg_end+1] = compensate_internal_states(
            x[:, seg_start:seg_end+1], -dx_smooth[:, seg_start:seg_end+1], quat[:, seg_start:seg_end+1]
        )

        # Save results
        zupt_ins_trajectory = Trajectory(
            t = inertial.t,
            pos = x[0:3, :],
            R_nb = orientation.euler_to_matrix(x[6:9, :])
        )

        # ------------------------------------------------------------------ #
        # Miscellaneous / prepare next segment
        # ------------------------------------------------------------------ #
        dx[:, seg_end]        = 0.0
        P[0:2, 8, seg_end]    = 0.0
        P[8, 0:2, seg_end]    = 0.0

        if seg_end != N - 1:
            seg_start = seg_end + 1
            seg_end   = N - 1
        else:
            break

    return zupt, zupt_ins_trajectory, step_seg

def initialize_nav(u:NDArray, init_heading: float, init_pos: NDArray)->Tuple[NDArray,NDArray]:
    """
    Calculate the initial state of the navigation equations.

    Parameters
    ----------
    u : ndarra, shape (6, N)
        IMU data matrix; rows are [ax, ay, az, wx, wy, wz], columns are samples.
    init_heading : float
        Initial heading (yaw) in radians, from settings.
    init_pos : ndarray, shape (3,)
        Initial position [x, y, z], from settings.

    Returns
    -------
    x : ndarray, shape (9,)
        Initial navigation state vector [position (3), velocity (3), euler angles (3)].
    quat : ndarray, shape (4,)
        Quaternion representing the initial attitude of the platform.

    Notes
    -----
    Assumes the system is stationary during the first 20 samples.
    Roll and pitch are estimated from the mean of the first 20 accelerometer readings.
    """
    u = np.asarray(u, dtype=float)

    # Estimate roll and pitch from mean of first 20 accelerometer readings
    f_u = np.mean(u[0, :20])
    f_v = np.mean(u[1, :20])
    f_w = np.mean(u[2, :20])

    roll  = np.arctan2(-f_v, -f_w)
    pitch = np.arctan2(f_u, np.sqrt(f_v**2 + f_w**2))

    attitude = np.array([roll, pitch, init_heading])

    # Compute initial quaternion from attitude
    R_bn = orientation.euler_to_matrix(attitude)
    quat = orientation.dcm2q(R_bn)

    # Assemble initial state vector
    x = np.zeros(9)
    x[0:3] = init_pos
    x[6:9] = attitude

    return x, quat


def init_filter(simdata: INSConfig)->Tuple[NDArray, NDArray, NDArray]:
    """
    Initialize the Kalman filter matrices.

    Parameters
    ----------
    sigma_acc : array_like, shape (3,)
        Standard deviations of accelerometer noise.
    sigma_gyro : array_like, shape (3,)
        Standard deviations of gyroscope noise.
    sigma_vel : array_like, shape (3,)
        Standard deviations of velocity measurement noise.

    Returns
    -------
    Q : ndarray, shape (6, 6)
        Process noise covariance matrix.
    R : ndarray, shape (3, 3)
        Measurement noise covariance matrix.
    H : ndarray, shape (3, 9)
        Observation matrix.
    """
    sigma_acc  = simdata.sigma_acc_array
    sigma_gyro = simdata.sigma_gyro_array 
    sigma_vel  = simdata.sigma_vel_array

    # Process noise covariance matrix
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = np.diag(sigma_acc**2)
    Q[3:6, 3:6] = np.diag(sigma_gyro**2)

    # Observation matrix — maps velocity states (indices 3:6) to measurements
    H = np.zeros((3, 9))
    H[0:3, 3:6] = np.eye(3)

    # Measurement noise covariance matrix
    R = np.diag(sigma_vel**2)

    return Q, R, H

def compensate_internal_states(x_in: NDArray, dx: NDArray, q_in: NDArray)-> Tuple[NDArray, NDArray]:
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

def state_matrix(q: NDArray, u: NDArray, Ts: float)-> Tuple[NDArray, NDArray]:
    """
    Calculate the state transition matrix F and the process noise gain matrix G.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Quaternion representing current orientation [q0, q1, q2, q3].
    u : ndarray, shape (6,)
        IMU data: first 3 elements are specific force, last 3 are angular rates.
    Ts : float
        Sampling time (seconds).

    Returns
    -------
    F : ndarray, shape (9, 9)
        Discrete-time state transition matrix.
    G : ndarray, shape (9, 6)
        Discrete-time process noise gain matrix.
    """

    # Convert quaternion to rotation matrix (body-to-navigation)
    Rb2t = orientation.q2dcm(q)

    # Transform specific force from body frame to navigation frame
    f_t = Rb2t @ u[:3]

    # Skew-symmetric matrix of the specific force vector
    St = np.array([
        [ 0.0,    -f_t[2],  f_t[1]],
        [ f_t[2],  0.0,    -f_t[0]],
        [-f_t[1],  f_t[0],  0.0   ]
    ])

    O = np.zeros((3, 3))
    I = np.eye(3)

    # Continuous-time state transition matrix (9x9)
    Fc = np.block([
        [O, I, O ],
        [O, O, St],
        [O, O, O ]
    ])

    # Continuous-time process noise gain matrix (9x6)
    Gc = np.block([
        [O,     O    ],
        [Rb2t,  O    ],
        [O,     -Rb2t]
    ])

    # First-order discrete-time approximation
    F = np.eye(Fc.shape[0]) + Ts * Fc
    G = Ts * Gc

    return F, G

def state_matrix_closed_form(q: NDArray, u: NDArray, Ts: float)-> Tuple[NDArray, NDArray]:
    """
    Calculate the state transition matrix F and the process noise gain matrix G.

    Parameters
    ----------
    q : ndarray, shape (4,)
        Quaternion representing current orientation [q0, q1, q2, q3].
    u : ndarray, shape (6,)
        IMU data: first 3 elements are specific force, last 3 are angular rates.
    Ts : float
        Sampling time (seconds).

    Returns
    -------
    F : ndarray, shape (9, 9)
        Discrete-time state transition matrix.
    G : ndarray, shape (9, 6)
        Discrete-time process noise gain matrix.
    """

    # Convert quaternion to rotation matrix (body-to-navigation)
    R_nb = orientation.q2dcm(q)

    # Compute skew symmetric matrix of body frame measured accel
    a_skew = np.array([
        [ 0.0,    -u[2],  u[1]],
        [ u[2],  0.0,    -u[0]],
        [-u[1],  u[0],  0.0   ]
    ])

    # Compute skew symmetrix matrix of body frame angular rate
    w_skew = np.array([
        [ 0.0,    -u[5],  u[4]],
        [ u[5],  0.0,    -u[3]],
        [-u[4],  u[3],  0.0   ]
    ])

    # Compute norm of angular rate
    w_norm = linalg.norm(u[3:6])

    # Compute rotation matrix from angular rate
    R_delta = linalg.expm(Ts * w_skew)

    # Identity matrix for convenience
    I = np.eye(3)
    O = np.zeros((3, 3))

    phi_pv = I*Ts
    phi_oo = R_delta.T

    if w_norm > 1e-9 :
        
        # compute sum term    
        sum = -np.sum(np.array([
            np.linalg.matrix_power(-w_skew * Ts, int(ki)) / factorial(int(ki))
            for ki in range(3)
        ]))

        phi_po = -R_nb @ a_skew @ (I*Ts**2/2 - (R_delta.T - sum)/w_norm**2)
        phi_vo = -R_nb @ a_skew @ (I*Ts + (w_skew/w_norm**2)@(R_delta.T - I + w_skew*Ts))
    else :
        phi_po = -R_nb @ a_skew * Ts**2/2
        phi_vo = -R_nb @ a_skew * Ts

    # Discrete time dynamics
    Fx = np.block([
        [I, phi_pv, phi_po],
        [O, I, phi_vo],
        [O, O, phi_oo]
    ])


    # Discrete time control matrix
    Fu = Ts * np.block([
        [O,     O    ],
        [-R_nb,  O    ],
        [O,     -I]
    ])

    return Fx, Fu

def navigation_equations(x:NDArray, u:NDArray, q:NDArray, Ts:float, g:float)->Tuple[NDArray,NDArray]:
    """
    Mechanized navigation equations of the inertial navigation system.

    Parameters
    ----------
    x : ndarray, shape (9,)
        Old navigation state [position (3), velocity (3), euler angles (3)].
    u : ndarray, shape (6,)
        IMU data: first 3 elements are specific force, last 3 are angular rates.
    q : ndarray, shape (4,)
        Old quaternion [q0, q1, q2, q3] (scalar-first).
    Ts : float
        Sampling time (seconds).
    g : float
        Gravitational acceleration (m/s^2).

    Returns
    -------
    y : ndarray, shape (9,)
        New navigation state [position (3), velocity (3), euler angles (3)].
    q : ndarray, shape (4,)
        Updated quaternion.

    Notes
    -----
    This mechanization is simplified — several higher-order terms are neglected.
    It is intended for low-cost sensors with moderate velocities only.
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float).copy()
    u = np.asarray(u, dtype=float)

    y = np.zeros_like(x)

    # ------------------------------------------------------------------ #
    # Update quaternion given angular rate measurements
    # ------------------------------------------------------------------ #
    w_tb = u[3:6]
    P, Q, R = w_tb * Ts  # scaled angular increments

    OMEGA = 0.5 * np.array([
        [ 0,  R, -Q,  P],
        [-R,  0,  P,  Q],
        [ Q, -P,  0,  R],
        [-P, -Q, -R,  0]
    ])

    v = np.linalg.norm(w_tb) * Ts
    if v != 0:
        q = (np.cos(v / 2) * np.eye(4) + (2 / v) * np.sin(v / 2) * OMEGA) @ q
        q /= np.linalg.norm(q)

    # ------------------------------------------------------------------ #
    # Extract Euler angles from updated quaternion
    # ------------------------------------------------------------------ #
    Rb2t = orientation.q2dcm(q)

    y[6] = np.arctan2(Rb2t[2, 1], Rb2t[2, 2])                                      # roll
    y[7] = np.arctan2(-Rb2t[2, 0], np.sqrt(Rb2t[2, 1]**2 + Rb2t[2, 2]**2))        # pitch
    y[8] = np.arctan2(Rb2t[1, 0], Rb2t[0, 0])                                      # yaw

    # ------------------------------------------------------------------ #
    # Update position and velocity
    # ------------------------------------------------------------------ #
    g_t = np.array([0.0, 0.0, g])

    f_t   = Rb2t @ u[:3]           # specific force in navigation frame
    acc_t = f_t + g_t              # remove gravity to get acceleration

    A = np.eye(6)
    A[0, 3] = Ts
    A[1, 4] = Ts
    A[2, 5] = Ts

    B = np.vstack([(Ts**2 / 2) * np.eye(3),
                   Ts          * np.eye(3)])

    y[:6] = A @ x[:6] + B @ acc_t

    return y, q


if __name__ == "__main__":
    inertial = InertialData.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    simdata = INSConfig(segmentation_thrsld=0.03)
    
    zupt, ins_traj, segs = smoothed_zupt_aided_ins(inertial, simdata)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.grid(visible=True)
    # ax.plot(inertial.t, (inertial.u[1:3,:]**2).sum(axis=0), linewidth=0.5)
    # ax.scatter(inertial.t[segs], 100*np.ones_like(segs), marker='x', c='r', s=5)



