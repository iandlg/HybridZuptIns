import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import Sequence, Any, Tuple, Dict
from dataclasses import dataclass

import src.offline_correction.hsgp as hsgp
import src.online_correction.kalman_filter as kf
from src.zupt_ins.data_classes import InertialData, Trajectory
from src.zupt_ins.initialization import INSConfig
import src.zupt_ins.detector as detector
from src.zupt_ins.zupt_ins import (
    init_filter, initialize_nav, navigation_equations, state_matrix, StepDetector,
    compensate_internal_states
)
import src.zupt_ins.orientation as orientation

@dataclass
class GPparameters :
    hyperparameters: Dict[str, NDArray] # contains the hyperparameters for each output type ("yaw", "pos_0" ...)
    feature_dim: int
    m : int
    feature_std: float = 1.0
    feature_mean: float = 0.0


def hybrid_zupt_aided_ins(
        inertial: InertialData,
        simdata: INSConfig,
        gt_traj: Trajectory,
        gp_params : GPparameters
    ) -> Tuple[NDArray, Trajectory, Sequence[int]]:
    """
    Run the open-loop zero-velocity aided INS Kalman filter with RTS smoothing.

    Parameters
    ----------

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

    # Check the ground truth is aligned with the inertial timesteps
    if not TimeSeries.is_compatible(inertial, gt_traj):
        raise ValueError("TimeSeries need to be aligned.")
    
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

    # Initialize HSGP
    beta = np.zeros(())
    P_beta = np.zeros()

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

        # ------------------------------------------------------------------ #
        # GP correction
        # ------------------------------------------------------------------ #



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




if __name__ == "__main__":
    from src.zupt_ins.data_classes import TimeSeries
    from src.config.results_io import ResultsSaver
    from src.offline_correction.gp import hyperparameters_from_csv

    config = ResultsSaver.load_json(
        PROJECT_ROOT / "src/config/online_correction_configs/hybrid_zins.json"
    )

    # Load hyperparameters from variability results
    hyperparameters = hyperparameters_from_csv(PROJECT_ROOT / config["hyperparameter_path"])

    data_path = PROJECT_ROOT / config["data_path"]
    trial_id = config["trial_id"]
    sim_config = INSConfig()

    gp_config = GPparameters(
        hyperparameters=hyperparameters,
        m=config["gp_parameters"]["m"],
        feature_dim=config["gp_parameters"]["feature_mean"],
        feature_mean=config["gp_parameters"]["feature_mean"],
        feature_std=config["gp_parameters"]["feature_standard_deviation"]
    )

    # Load data
    inertial = InertialData.from_csv_int(data_path, trial_id)
    gt_traj = Trajectory.from_csv_int(data_path, trial_id)

    # Truncate to overlapping time window and align ground truth to IMU timestamps
    inertial_trunc, gt_traj_trunc = TimeSeries.truncate_to_overlap(inertial, gt_traj)
    gt_traj_aligned = gt_traj_trunc.temporal_alignment(inertial_trunc.t)

    # Compute INS trajectory from inertial data
    zupt, ins_traj, segs = hybrid_zupt_aided_ins(
        inertial_trunc,
        sim_config,
        gt_traj_aligned,
        gp_config,
    )
    