from typing import Tuple, List
from numpy.typing import NDArray
import numpy as np

from src.zupt_ins.initialization import INSConfig
from src.zupt_ins.zupt_ins import smoothed_zupt_aided_ins
from src.zupt_ins.data_classes import InertialData, Trajectory, TimeSeries
from src.zupt_ins.trajectory_transform import transform_position, transform_orientation
import src.zupt_ins.orientation as orientation


def compute_aligned_ins_trajectory(
    data_path,
    trial_id: int,
    sim_config: INSConfig = INSConfig(),
    orientation_offset: NDArray = np.zeros(3),
)-> Tuple[Trajectory, Trajectory, NDArray, List[int], InertialData, INSConfig]:
    """
    Load inertial and ground truth data, compute an INS trajectory,
    and align it to the ground truth.

    Parameters
    ----------
    data_path : Path or str
        Path to the data directory.
    trial_id : int
        Trial/session identifier passed to the CSV loaders.
    sim_config : INSConfig,
        INS configuration. Defaults to INSConfig().
    orientation_offset : NDArray,
        3-element orientation offset for transform_orientation.
        Defaults to np.zeros(3).

    Returns
    -------
    ins_traj_aligned : Trajectory
        The INS trajectory aligned to ground truth.
    gt_traj_aligned : Trajectory
        The ground truth trajectory aligned to the IMU time axis.
    zupt : array-like
        ZUPT detection signal.
    segs : any
        Segmentation output from smoothed_zupt_aided_ins.
    """
    # Load data
    inertial = InertialData.from_csv_int(data_path, trial_id)
    gt_traj = Trajectory.from_csv_int(data_path, trial_id)

    # Truncate to overlapping time window and align ground truth to IMU timestamps
    inertial_trunc, gt_traj_trunc = TimeSeries.truncate_to_overlap(inertial, gt_traj)
    gt_traj_aligned = gt_traj_trunc.temporal_alignment(inertial_trunc.t)

    # Compute INS trajectory from inertial data
    zupt, ins_traj, segs = smoothed_zupt_aided_ins(inertial_trunc, sim_config)

    # # Compare sources of position
    # pos_x = x_end[0:3]
    # pos_trj = ins_traj.pos[:,-1]

    # # Compare sources of velocity 
    # vel_x = x_end[3:6]
    # vel_trj = ins_traj.vel[:,-1] if ins_traj.vel is not None else None

    # # Compare sources of orientation
    # R_nb_end = ins_traj.R_nb[:,:,-1]
    # R_x = orientation.euler_to_matrix(x_end[6:9])
    # R_q = orientation.q2dcm(quat_end)

    # Last index to use for calibration: first point > 3m from start.
    distances = np.sqrt(np.sum((ins_traj.pos[:, 0:1] - ins_traj.pos) ** 2, axis=0))[segs]
    b = segs[np.argmax(distances > sim_config.calibration_distance_m)]

    # Calibration indices, excluding ZUPT frames.
    calib_idxs = np.array([i for i in range(0, b + 1) if not zupt[i]])

    # Rigidly align position and orientation to ground truth
    ins_traj_aligned, R_nprime_n, t = transform_position(ins_traj, gt_traj_aligned, calib_idxs)
    # print(x_end.shape)
    # x_end[0:3] = (R_nprime_n @ x_end[0:3, None] + t).flatten()
    # x_end[3:6] = (R_nprime_n @ x_end[3:6, None]).flatten()
    # R_end_nprime_b = R_nprime_n @ orientation.q2dcm(quat_end) 
    ins_traj_aligned, R_b_bprime = transform_orientation(ins_traj_aligned, gt_traj_aligned, zupt, orientation_offset, calib_idxs)
    # R_end_nprime_bprime =  R_end_nprime_b @ R_b_bprime
    # quat_end = orientation.dcm2q(R_end_nprime_bprime)
    # x_end[6:9] = orientation.matrix_to_euler(R_end_nprime_bprime)

    # Update inertial data 
    inertial = InertialData(
        inertial_trunc.t,
        u = np.vstack([
            R_b_bprime.T @ inertial_trunc.u[0:3,:],
            R_b_bprime.T @ inertial_trunc.u[3:6, :]
        ])
    )
    sim_config.g = R_nprime_n @ np.array([0,0,sim_config.g])

    return ins_traj_aligned, gt_traj_aligned, zupt, segs, inertial, sim_config
