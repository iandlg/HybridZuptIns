import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from enum import Enum
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.spatial.transform import Rotation

from src.zupt_ins.data_classes import Trajectory, TimeSeries

class LocalFrame(Enum):
    BODY = 1
    HEADING = 2

def compute_training_io(
        traj: Trajectory,
        traj_gt: Trajectory,
        step_seg: List[int],
        ref_frame: LocalFrame = LocalFrame.BODY
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating]
    ]:
    """
    Compute training inputs and outputs for GP regression from a pair of trajectories.

    Computes the yaw difference and position correction between the inertial
    trajectory and ground truth at each step segment, expressed in the specified
    local reference frame.

    Parameters
    -------
    traj : Trajectory
        Inertial trajectory, temporally aligned with traj_gt.
    traj_gt : Trajectory
        Ground truth trajectory, temporally aligned with traj.
    step_seg : List[int]
        Indices marking step boundaries in the trajectory.
    ref_frame : LocalFrame, optional
        Reference frame for step vector computation. Default is LocalFrame.BODY.

    Returns
    -------
    output_yawdiff : NDArray[np.floating], shape (n_steps - 1,)
        Per-step yaw difference between ground truth and inertial trajectory.
    output_pos : NDArray[np.floating], shape (3, n_steps - 1)
        Per-step position correction (ground truth minus inertial) in ref_frame.
    input_feature : NDArray[np.floating], shape (3, n_steps - 1)
        Inertial step vectors in ref_frame, used as regression inputs.

    Raises
    ------
    ValueError
        If traj and traj_gt are not temporally compatible.
    """
    # Check if time series are overlapping and sampled at the same time 
    if not TimeSeries.is_compatible(traj, traj_gt) :
        raise ValueError(f"TimeSeries must be compatible.")

    inertial_euler = traj.euler_nb.T[step_seg,:] # (n_steps,3)
    gt_euler = traj_gt.euler_nb.T[step_seg,:]    # (n_steps,3)

    # Compute yaw training output
    output_yawdiff = np.diff(       # (n_steps - 1,)
        np.unwrap(
            gt_euler[:,2]
        )
    ) - np.diff(
        np.unwrap(
            inertial_euler[:,2]
        )
    )
    
    # Compute Step vectors in specified reference frame
    funs = {
        LocalFrame.BODY : Trajectory.step_vectors_body,
        LocalFrame.HEADING : Trajectory.step_vectors_heading
    }

    input_feature = funs[ref_frame](traj, step_seg)
    gt_steps = funs[ref_frame](traj_gt, step_seg)

    output_pos = gt_steps - input_feature

    print(f"{input_feature.shape = }")
    print(f"{output_yawdiff.shape = }")
    print(f"{output_pos.shape = }")
    return output_yawdiff, output_pos, input_feature

def compute_corrections(x: NDArray[np.floating], y: NDArray[np.floating]):
    """
    Estimate output corrections using leave-one-fold-out Gaussian Process regression.

    Performs 10-fold cross-validation, fitting a GP with an RBF + white noise
    kernel on each fold. Also computes a static (mean) correction as a baseline.

    Parameters
    -------
    x : NDArray[np.floating], shape (n_features, n_samples)
        Input features, one column per sample.
    y : NDArray[np.floating], shape (n_samples,)
        Scalar target values to regress.

    Returns
    -------
    y_testing_GP : NDArray[np.floating], shape (n_samples,)
        GP-predicted corrections assembled across all folds.
    y_testing_static : NDArray[np.floating], shape (n_samples,)
        Static correction, equal to the global mean of y for all samples.
    """
    n_samples = x.shape[1]
    y_testing_GP = np.zeros(n_samples)
    D = 10

    kernel = RBF() + WhiteKernel()  # RBF signal + noise term

    for i in range(1, D + 1):
        # --- indices --------------------------------------------------------
        test_start = int(np.floor(n_samples / D * (i - 1)))
        test_end   = int(np.floor(n_samples / D * i))

        training_ind = list(range(0, test_start)) + list(range(test_end, n_samples))
        testing_ind  = list(range(test_start, test_end))

        # --- data -----------------------------------------------------------
        x_train = x[:, training_ind].T   # (n_train, n_features)
        y_train = y[training_ind]        # (n_train, )
        x_test  = x[:, testing_ind].T    # (n_test,  n_features)

        # --- fit + predict --------------------------------------------------
        model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
        model.fit(x_train, y_train)
        y_testing_GP[testing_ind] = model.predict(x_test)

    # --- static correction --------------------------------------------------
    y_testing_static = np.full(n_samples, float(np.mean(y)))

    return y_testing_GP, y_testing_static

def apply_corrections(
    traj: Trajectory,
    yawdiff_correction: NDArray[np.floating],
    pos_correction: NDArray[np.floating],
    segs: List[int],
    ref_frame: LocalFrame = LocalFrame.BODY
) -> Trajectory:
    """
    Apply yaw and position corrections to produce a corrected step-wise trajectory.

    Integrates the corrected yaw increments to obtain a new heading sequence,
    then re-integrates step vectors in the specified reference frame to produce
    corrected positions. The output trajectory is defined only at step segment
    indices.

    Parameters
    -------
    traj : Trajectory
        Complete inertial trajectory to correct.
    yawdiff_correction : NDArray[np.floating], shape (n_steps - 1,)
        Regressed per-step yaw correction (radians).
    pos_correction : NDArray[np.floating], shape (3, n_steps - 1)
        Regressed per-step x, y, z position corrections (metres).
    segs : List[int]
        Indices marking step boundaries in traj.
    ref_frame : LocalFrame, optional
        Reference frame used for step vector computation. Default is LocalFrame.BODY.

    Returns
    -------
    Trajectory
        Corrected trajectory sampled at the step segment indices, with updated
        positions and orientations.
    """
    
    euler = traj.euler_nb[:, segs]  # (3, n_steps) or n_steps-1 ???????

    # --- compute corrected yaws ---------------------------------------------
    diff_yaw = np.diff(euler[2, :]) + yawdiff_correction
    new_yaws = np.cumsum(np.concatenate([[euler[2, 0]], diff_yaw]))
    new_yaws = (new_yaws + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    new_euler = euler.copy()
    new_euler[2, :] = new_yaws
    new_R = np.asarray(Rotation.from_euler('xyz', new_euler.T).as_matrix()).transpose(1,2,0)

    # --- integrate corrected positions --------------------------------------
    n = len(segs)
    pos_out = np.zeros((3, n))
    pos_out[:, 0] = traj.pos[:, segs[0]]

    # Compute Step vectors in specified reference frame
    funs = {
        LocalFrame.BODY : Trajectory.step_vectors_body,
        LocalFrame.HEADING : Trajectory.step_vectors_heading
    }

    steps = funs[ref_frame](traj, segs)

    for k in range(1, n):
        pos_out[:, k] = new_R[:,:,k-1] @ (steps[:,k-1] + pos_correction[:, k - 1]) + pos_out[:, k - 1]

    # --- build output trajectory --------------------------------------------
    return Trajectory(
        t   = traj.t[segs],
        pos = pos_out,
        R_nb = traj.R_nb[:, :, segs],  # carry corrected R if available, else original
    )

if __name__ == "__main__" :
    from src.zupt_ins.initialization import INSConfig
    from src.zupt_ins.zupt_ins import smoothed_zupt_aided_ins
    from src.zupt_ins.data_classes import InertialData
    from src.zupt_ins.trajectory_transform import transform_position, transform_orientation

    # Load data
    inertial = InertialData.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    gt_traj = Trajectory.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    simdata = INSConfig()

    # Data preprocessing to fit gt to imu data
    inertial_trunc, gt_traj_trunc = TimeSeries.truncate_to_overlap(inertial, gt_traj)
    gt_traj_aligned = gt_traj_trunc.temporal_alignment(inertial_trunc.t)

    # Compute trajectory from inertial data
    zupt, ins_traj, segs = smoothed_zupt_aided_ins(inertial_trunc, simdata)

    # Rigidly transform the positions and orientations of the computed trajectory
    ins_traj_aligned = transform_position(ins_traj, gt_traj_aligned, zupt)
    ins_traj_aligned = transform_orientation(ins_traj_aligned, gt_traj_aligned, zupt, np.zeros(3))

    # Compute inputs and outputs for regression
    output_yawdiff, output_pos, input_feature = compute_training_io(ins_traj_aligned, gt_traj_aligned, segs, ref_frame=LocalFrame.BODY)

    # Regress test outputs
    y_yaw_GP, y_yaw_static = compute_corrections(input_feature, output_yawdiff)

    y_pos_GP = np.empty(output_pos.shape)
    y_pos_static = np.empty(output_pos.shape)
    for d in range(3):
        y_pos_GP[d, :], y_pos_static[d, :] = compute_corrections(input_feature, output_pos[d, :])

    GP_step_traj = apply_corrections(ins_traj, y_yaw_GP, y_pos_GP, segs, ref_frame=LocalFrame.BODY)
    static_step_traj = apply_corrections(ins_traj, y_yaw_static, y_pos_static, segs, ref_frame=LocalFrame.BODY)
    GT_step_traj = gt_traj_aligned[segs]
    ins_step_traj = ins_traj_aligned[segs]

    import src.plotting.plot_trajectories as plot
    import matplotlib.pyplot as plt

    plot.plot_groundtruth_vs_inertial_positions([ins_step_traj, GP_step_traj, static_step_traj], GT_step_traj)
    plot.plot_groundtruth_vs_inertial_orientations(GP_step_traj, GT_step_traj)
    plot.plot_position_rmse([ins_step_traj, GP_step_traj,static_step_traj], GT_step_traj)
    plt.show()