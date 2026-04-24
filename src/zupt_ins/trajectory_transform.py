import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares


from src.zupt_ins import orientation
from src.zupt_ins.data_classes import  Trajectory
import src.zupt_ins.orientation as orientation

_CALIBRATION_DISTANCE = 3   # m; length of the trajectory to consider for calibration

def transform_trajectory(
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
    for k in range(ins_traj.R.shape[2]):
        ins_traj.R[:, :, k] = R @ ins_traj.R[:, :, k]

    return ins_traj


if __name__ == "__main__":
    pass