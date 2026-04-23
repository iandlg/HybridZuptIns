import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)


from scipy.spatial.transform import Rotation, Slerp
import numpy as np

from src.zupt_ins.data_classes import Trajectory

def temporal_alignment(inertial_t: np.ndarray, ground_truth: Trajectory) -> Trajectory:
    """
    Align ground truth trajectory to inertial measurement timestamps via interpolation.

    Position is interpolated linearly. Rotations are interpolated using SLERP
    (Spherical Linear Interpolation) on SO(3), which guarantees valid rotation
    matrices and follows the shortest geodesic path between orientations.

    Parameters
    ----------
    inertial_t : np.ndarray, shape (N,)
        Timestamps of the inertial measurements in seconds.
    ground_truth : Trajectory
        Ground truth trajectory, potentially sampled at a different rate or
        with a different time base than the inertial measurements.

    Returns
    -------
    Trajectory
        Ground truth trajectory resampled at the inertial timestamps,
        with the same time base as `inertial_t`.
    """
    t_gt  = ground_truth.t
    pos_gt = ground_truth.pos                        # (3, N)
    R_gt   = ground_truth.R.transpose(2, 0, 1)       # (N, 3, 3)

    # --- zero-order-hold extension on the left ---
    if inertial_t[0] < t_gt[0]:
        t_gt   = np.concatenate([[inertial_t[0]], t_gt])
        pos_gt = np.hstack([pos_gt[:, :1], pos_gt])
        R_gt   = np.concatenate([R_gt[:1], R_gt], axis=0)
    # print(f"{t_gt.shape = }")
    # print(f"{pos_gt.shape = }")
    # print(f"{R_gt.shape = }")
    # --- zero-order-hold extension on the right ---
    if inertial_t[-1] > t_gt[-1]:
        t_gt   = np.concatenate([t_gt, [inertial_t[-1]]])
        pos_gt = np.hstack([pos_gt, pos_gt[:, -1:]])
        R_gt   = np.concatenate([R_gt, R_gt[-1:]], axis=0)
    # print(f"{t_gt.shape = }")
    # print(f"{pos_gt.shape = }")
    # print(f"{R_gt.shape = }")

    # Interpolate position
    pos = np.vstack([
        np.interp(inertial_t, t_gt, pos_gt[i])
        for i in range(3)
    ])

    # SLERP for rotations — (N, 3, 3) expected by Rotation
    slerp = Slerp(t_gt, Rotation.from_matrix(R_gt, assume_valid=False))
    R = slerp(inertial_t).as_matrix().transpose(1, 2, 0)  # back to (3, 3, N)

    return Trajectory(t=inertial_t, pos=pos, R=R)

if __name__=="__main__" :
    from src.zupt_ins.data_classes import InertialData

    imu = InertialData.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    gt = Trajectory.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)

    print(f"{imu.t = }")
    print(f"{gt.t = }")

    gt_aligned = temporal_alignment(imu.t, gt)
