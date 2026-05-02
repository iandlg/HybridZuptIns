from typing import Optional
from numpy.typing import NDArray
import numpy as np

from src.zupt_ins.initialization import INSConfig
from src.zupt_ins.zupt_ins import smoothed_zupt_aided_ins
from src.zupt_ins.data_classes import InertialData, ReferenceFrame, Trajectory, TimeSeries
from src.zupt_ins.trajectory_transform import transform_position, transform_orientation


def compute_aligned_ins_trajectory(
    data_path,
    trial_id: int,
    sim_config: INSConfig = INSConfig(),
    orientation_offset: NDArray = np.zeros(3),
):
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

    # Rigidly align position and orientation to ground truth
    ins_traj_aligned = transform_position(ins_traj, gt_traj_aligned, zupt)
    ins_traj_aligned = transform_orientation(ins_traj_aligned, gt_traj_aligned, zupt, orientation_offset)

    return ins_traj_aligned, gt_traj_aligned, zupt, segs
