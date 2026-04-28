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

    # Check if time series are overlapping and sampled at the same time 
    if not TimeSeries.is_compatible(traj, traj_gt) :
        raise ValueError(f"TimeSeries must be compatible.")

    inertial_euler = traj.euler_nb.T[step_seg,:] # (N,3)
    gt_euler = traj_gt.euler_nb.T[step_seg,:]    # (N,3)

    # Compute yaw training output
    output_yawdiff = np.diff(       # (N,)
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
    return output_yawdiff, output_pos, input_feature

def compute_corrections(x: NDArray[np.floating], y: NDArray[np.floating]):
    """
    x : (n_features, n_samples)
    y : (1, n_samples)

    Returns
    -------
    y_testing_GP     : (n_samples,)
    y_testing_static : (n_samples,)
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
    Parameters
    -------
    traj : Trajectory,
        Complete trajectory to correct
    yawidff_correction : NDArray[np.floating], shape (n_steps,)
        Regressed yaw correction
    pos_correction : NDArray[np.floating], shape (n_steps, 3)
        Regressed x,y,z corrections
    segs : List[int]
    """
    
    euler = traj.euler_nb[:, segs]  # (3, n_steps) or n_steps-1 ???????

    # --- compute corrected yaws ---------------------------------------------
    diff_yaw = np.diff(euler[2, :]) + yawdiff_correction
    new_yaws = np.cumsum(np.concatenate([[euler[0, 2]], diff_yaw]))
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
    GT_step_traj = gt_traj_aligned[segs]
    ins_step_traj = ins_traj_aligned[segs]

    import src.plotting.plot_trajectories as plot
    import matplotlib.pyplot as plt

    plot.plot_groundtruth_vs_inertial_positions([ins_step_traj, GP_step_traj], GT_step_traj)
    plot.plot_groundtruth_vs_inertial_orientations(GP_step_traj, GT_step_traj)
    plot.plot_position_rmse([ins_step_traj, GP_step_traj], GT_step_traj)
    plt.show()