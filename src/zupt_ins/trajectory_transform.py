import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from typing import List

from src.zupt_ins import orientation
from src.zupt_ins.data_classes import  Trajectory
import src.zupt_ins.orientation as orientation

_CALIBRATION_DISTANCE = 3   # m; length of the trajectory to consider for calibration

def transform_position(
        ins_traj:Trajectory,
        gt_traj:Trajectory,
        zupt: NDArray[np.bool],
    ):
    """
    Applies a rigid transformation to the INS trajectory to match the ground truth reference frame.
    IMPORTANT : Both TimeSeries must be overlapping

    Parameters
    ----------
    ins_traj : Trajectory,
        Trajectory computed from inertial data.
    gt_traj : Trajectory,
        Ground truth trajectory

    Returns
    -------
    ins_traj_aligned : Trajectory,
        Trajectory computed from inertial data aligned with GT reference.
    """

    # Last index to use for calibration: first point >3m from start.
    distances = np.sqrt(np.sum((ins_traj.pos[:, 0:1] - ins_traj.pos) ** 2, axis=0))
    b = np.argmax(distances > _CALIBRATION_DISTANCE)

    # Calibration indices, excluding ZUPT frames.
    indices = np.array([i for i in range(0, b + 1) if not zupt[i]])

    # Residual function: ground_truth is already interpolated at inertial instants.
    def residuals(x: NDArray[np.floating]):
        R = orientation.euler_to_matrix(np.array([np.pi, 0, x[0]]))
        t = x[1:4, np.newaxis]
        diff = gt_traj.pos[:, indices] - (R @ ins_traj.pos[:3, indices] + t)
        return diff.ravel()

    x0 = np.zeros(4)
    result = least_squares(residuals, x0, method='lm')
    x = result.x

    # Apply the optimal rotation and translation.
    R = orientation.euler_to_matrix(np.array([np.pi, 0, x[0]]))
    t = x[1:4, np.newaxis]

    ins_traj.pos[:3, :] = R @ ins_traj.pos[:3, :] + t

    # Rotate all orientation matrices.
    for k in range(ins_traj.R_nb.shape[2]):
        ins_traj.R_nb[:, :, k] = R @ ins_traj.R_nb[:, :, k]

    return ins_traj


def euler_mse(
        angles: NDArray[np.floating],
        ins_traj: Trajectory,
        gt_traj: Trajectory,
        zupt: NDArray[np.bool],
        indices: List[int]
    ) -> np.ndarray:
    """
    Compute per-sample Euler angle residuals between inertial and ground truth.
    N is the number of time steps in the trajectory. 

    Parameters
    ---------
    angles : NDArray[np.floating], shape (3,)
        Angles being optimised
    ins_traj : Trajectory, 
        Original trajectory who's estimated orientations are being rotated
    gt_traj : Trajectory,
        Ground thruth trajectory for which to use reference frame
    zupt : NDArray[np.bool], shape (N,)
        True if foot in stance phase, used to calibrated pitch and roll.
    indices : List[int],
        Frames used to calibrate the yaw. 

    Assumptions:
        - Inertial and ground truth time series are already temporally aligned
          (i.e. index k in inertial corresponds to index k in ground_truth).
        - The two series fully overlap (no extrapolation needed).
    """
    N = len(ins_traj)

    # Apply trial rotation to every orientation matrix.
    R = orientation.euler_to_matrix(angles)
    R_rotated = np.einsum('...ij,jk->...ik', ins_traj.R_nb.transpose(2, 0, 1), R)  # (N,3,3)

    # Compute Euler angles for all frames at once.
    # eulers are shape (3,N)
    # R back to (3,3,N)
    ins_euler = orientation.matrix_to_euler(R_rotated.transpose(1,2,0)) 
    gt_euler = orientation.matrix_to_euler(gt_traj.R_nb)


    # Roll and pitch: computed over ZUPT frames.
    roll_res  = _wrapped_min_residuals(ins_euler[0, zupt], gt_euler[0, zupt])
    pitch_res = _wrapped_min_residuals(ins_euler[1, zupt], gt_euler[1, zupt])

    # Yaw: computed over the calibration window.
    yaw_res = _wrapped_min_residuals(ins_euler[2, indices], gt_euler[2, indices])

    return np.concatenate([roll_res, pitch_res, yaw_res])


def transform_orientation(
        ins_traj: Trajectory,
        gt_traj: Trajectory,
        zupt: NDArray[np.bool],
        initial_value: NDArray[np.floating]
    ) -> Trajectory:
    """
    Optimise the IMU orientation and temporal alignment against ground truth.

    Assumptions:
        - Inertial and ground truth time series are already temporally aligned
          (i.e. index k in inertial corresponds to index k in ground_truth).
        - The two series fully overlap (no extrapolation needed).
    """
    # Find the first overlapping index.
    a = 0

    # Last calibration index: first point more than 3 m from the start.
    distances = np.sqrt(np.sum((ins_traj.pos[:, 0:1] - ins_traj.pos) ** 2, axis=0))
    b = np.argmax(distances > _CALIBRATION_DISTANCE)

    # Calibration indices, excluding ZUPT frames.
    indices = np.array([i for i in range(a, b + 1) if not zupt[i]])

    # Optimise rotation to minimise Euler angle MSE.
    result = least_squares(euler_mse, initial_value, method='lm',
                           args=(ins_traj, gt_traj, zupt, indices))
    R = orientation.euler_to_matrix(result.x)

    # Apply optimal rotation to all orientation matrices (post-multiply).
    new_ins_R = np.einsum('ijk,kl->ijl', ins_traj.R_nb.transpose(2, 0, 1), R).transpose(1, 2, 0)

    # Return rotated orientations
    return Trajectory(
        t=ins_traj.t,
        pos=ins_traj.pos,
        R_nb=new_ins_R
    )

def _wrapped_min_residuals(ins_vals, gt_vals):
    """Minimum absolute residual across 0, +2π, -2π wrappings."""
    diff = ins_vals - gt_vals
    return np.min(np.abs([diff, diff + 2 * np.pi, diff - 2 * np.pi]), axis=0)


if __name__ == "__main__":
    pass