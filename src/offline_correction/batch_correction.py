import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

import src.zupt_ins.orientation as orientation 
from src.zupt_ins.data_classes import Trajectory, TimeSeries, ReferenceFrame


def compute_training_io(
        traj: Trajectory,
        traj_gt: Trajectory,
        step_seg: List[int],
        ref_frame: ReferenceFrame = ReferenceFrame.BODY
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

    inertial_euler = traj.euler_nb[:, step_seg] # (3,n_steps)
    gt_euler = traj_gt.euler_nb[:, step_seg]    # (3,n_steps)

    # Compute yaw training output
    output_yawdiff = np.diff(       # (n_steps - 1,)
        np.unwrap(gt_euler[2,:])
    ) - np.diff(
        np.unwrap(inertial_euler[2,:])
    )
    
    # Compute Step vectors in specified reference frame
    funs = {
        ReferenceFrame.BODY : Trajectory.step_vectors_body,
        ReferenceFrame.HEADING : Trajectory.step_vectors_heading
    }

    input_feature = funs[ref_frame](traj, step_seg)
    gt_steps = funs[ref_frame](traj_gt, step_seg)

    output_pos = gt_steps - input_feature

    return output_yawdiff, output_pos, input_feature



def compute_static_corrections(
        x: NDArray[np.floating],
        y: NDArray[np.floating],
    ) -> NDArray:
    n_samples = x.shape[1]
    return np.full(n_samples, float(np.mean(y)))

def apply_corrections(
    traj: Trajectory,
    yawdiff_correction: NDArray[np.floating],
    pos_correction: NDArray[np.floating],
    segs: List[int],
    ref_frame: ReferenceFrame = ReferenceFrame.BODY
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
    print(ref_frame)    
    euler = traj.euler_nb[:, segs]  # (3, n_steps)

    # --- compute corrected yaws ---------------------------------------------
    diff_yaw = np.diff(euler[2, :]) + yawdiff_correction
    new_yaws = np.cumsum(np.concatenate([[euler[2, 0]], diff_yaw]))

    new_euler = {
        ReferenceFrame.BODY: euler.copy(),
        ReferenceFrame.HEADING: np.zeros_like(euler),
    }.get(ref_frame) 

    if new_euler is None:
        raise ValueError(f"Unsupported Reference Frame: {str(ref_frame)}")

    new_euler[2, :] = new_yaws
    new_R = orientation.euler_to_matrix(new_euler)

    # --- integrate corrected positions --------------------------------------
    n = len(segs)
    pos_out = np.zeros((3, n))
    pos_out[:, 0] = traj.pos[:, segs[0]]

    # Compute Step vectors in specified reference frame
    funs = {
        ReferenceFrame.BODY : Trajectory.step_vectors_body,
        ReferenceFrame.HEADING : Trajectory.step_vectors_heading
    }

    steps = funs[ref_frame](traj, segs)

    for k in range(1, n):
        pos_out[:, k] = new_R[:,:,k-1] @ (steps[:,k-1] + pos_correction[:, k - 1]) + pos_out[:, k - 1]

    # Final R_nb rotation
    new_euler = euler.copy()
    new_euler[2, :] = new_yaws
    R_nb_new = orientation.euler_to_matrix(new_euler)

    # --- build output trajectory --------------------------------------------
    return Trajectory(
        t   = traj.t[segs],
        pos = pos_out,
        R_nb = R_nb_new,
    )



if __name__ == "__main__" :
    import src.zupt_ins.pipeline as pipeline
    import matplotlib.pyplot as plt
    import src.plotting.plot_trajectories as plot_traj
    import src.plotting.plot_corrections as plot_corr
    import src.offline_correction.hyperparameter_variability as variability
    import src.offline_correction.gp as gp
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Kernel
    from src.config.results_io import save_batch_correction_run, ResultsSaver
    
    config = ResultsSaver.load_json(
        PROJECT_ROOT / "src/config/batch_correction_configs/gp_kernel_parameters_body_dev.json"
    )

    data_path = PROJECT_ROOT / config["data_path"]
    trial_id = config["trial_id"]
    FRAME = ReferenceFrame(config["local_reference_frame"])

    n_restart_optimizer = config["n_restart_optimizer"]
    opt_parameters = config["optimization_parameters"]

    # Compute INS trajectory
    ins_traj_aligned, gt_traj_aligned, zupt, segs = pipeline.compute_aligned_ins_trajectory(
        data_path=data_path,
        trial_id=trial_id,
    )

    # Compute inputs and outputs for regression
    output_yawdiff, output_pos, input_feature = compute_training_io(
        ins_traj_aligned, gt_traj_aligned, segs, ref_frame=FRAME)

    # Regress test outputs and optimize 
    hyperparams = {}

    print(f"--- Computing dimension yaw corrections ---")
    yaw_kernel = (
        ConstantKernel(
                constant_value=opt_parameters["yaw"]["var_f"]["init"],
                constant_value_bounds=opt_parameters["yaw"]["var_f"]["bounds"]
            ) * RBF(
                length_scale=opt_parameters["yaw"]["ls"]["init"],
                length_scale_bounds=opt_parameters["yaw"]["ls"]["bounds"],
            )
            + WhiteKernel(
                noise_level=opt_parameters["yaw"]["var_f"]["init"],
                noise_level_bounds=opt_parameters["yaw"]["var_n"]["bounds"]
            )
    )
    y_yaw_gp, hyperparams["yaw"]= gp.compute_gp_corrections(
        input_feature, output_yawdiff,yaw_kernel, n_restarts_optimizer=n_restart_optimizer)
    y_yaw_static = compute_static_corrections(input_feature, output_yawdiff)

    y_pos_gp = np.empty(output_pos.shape)
    y_pos_static = np.empty(output_pos.shape)
    for d in range(3):
        print(f"--- Computing dimension {d} corrections ---")
        pos_key = f"pos_{d}"
        pos_kernel = (
            ConstantKernel(
                    constant_value=opt_parameters[pos_key]["var_f"]["init"],
                    constant_value_bounds=opt_parameters[pos_key]["var_f"]["bounds"]
                ) * RBF(
                    length_scale=opt_parameters[pos_key]["ls"]["init"],
                    length_scale_bounds=opt_parameters[pos_key]["ls"]["bounds"],
                )
                + WhiteKernel(
                    noise_level=opt_parameters[pos_key]["var_f"]["init"],
                    noise_level_bounds=opt_parameters[pos_key]["var_n"]["bounds"]
                )
        )
        y_pos_gp[d, :], hyperparams[pos_key] = gp.compute_gp_corrections(
            input_feature, output_pos[d, :], pos_kernel, n_restarts_optimizer=n_restart_optimizer)
        y_pos_static[d, :] = compute_static_corrections(input_feature, output_pos[d, :])

    gp.hyperparameters_to_csv(
        hyperparams, PROJECT_ROOT / f"out/hyperparameters/python/hyperparam_{config["trial_id"]}_f{config["local_reference_frame"]}.csv"
    )

    # # Regress test outputs with step time feature added
    # print(f"--- Computing dimension yaw corrections ---")
    # delta_t = np.diff(np.array(segs))/100
    # augmen_feature = np.empty((4, len(segs)-1))
    # augmen_feature[0:3, :] = input_feature
    # augmen_feature[3, :] = delta_t
    # kernel = (
    #     ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))
    #     * RBF(length_scale=np.ones(4), length_scale_bounds=(1e-4, 1e2))
    #     + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-9, 1e-1))
    # )

    # y_yaw_GP_aug, _, hyperparams["yaw_dt"] = compute_corrections(augmen_feature, output_yawdiff, kernel)

    # y_pos_GP_aug = np.empty(output_pos.shape)
    # for d in range(3):
    #     print(f"--- Computing dimension {d} corrections ---")
    #     y_pos_GP_aug[d, :], _, hyperparams[f"pos_{d}_dt"] = compute_corrections(augmen_feature, output_pos[d, :],kernel)
    
    # Compute stepwise trajectories
    gt_step_traj = gt_traj_aligned[segs]
    ins_step_traj = ins_traj_aligned[segs]
    static_step_traj = apply_corrections(ins_traj_aligned, y_yaw_static, y_pos_static, segs, ref_frame=FRAME)
    gp_step_traj = apply_corrections(ins_traj_aligned, y_yaw_gp, y_pos_gp, segs, ref_frame=FRAME)

    static_rmse = static_step_traj.rmse(gt_step_traj)
    ins_rmse    = ins_step_traj.rmse(gt_step_traj)

    trajs = {
        "model" : ins_step_traj,
        "model + static" : static_step_traj,
        "model + GP" : gp_step_traj,
        # "model + GP + dt" : GP_step_traj_aug
    }

    rmse_results = {}
    for key, val in trajs.items():
        rmse_results[key] = val.rmse(gt_step_traj)

    save_batch_correction_run(
        PROJECT_ROOT / "out/runs/offline_correction/batch",
        data_path=data_path,
        trial_id=trial_id,
        local_reference_frame=str(FRAME),
        n_restart_optimizer=n_restart_optimizer,
        rmse_results=rmse_results,
        opt_parameters=opt_parameters
    )

    rmse_per_fold, corrected_trajs = variability.evaluate_hyperparameter_variability(
        ins_traj_aligned, gt_traj_aligned, segs, hyperparams, ref_frame=FRAME,
        output_filename= PROJECT_ROOT / "out/hyperparameters/python/hparam_variability_results.csv"
    )

    plot_corr.plot_regression_results(
        output_yawdiff, y_yaw_static, y_yaw_gp, output_pos, y_pos_static, y_pos_gp)
    variability.plot_hyperparameter_rmse_variability(
        rmse_per_fold, rmse_results["model"], rmse_results["model + static"])
    plot_traj.plot_groundtruth_vs_inertial_positions(trajs, gt_step_traj[:20])
    plot_traj.plot_groundtruth_vs_inertial_orientations(trajs, gt_step_traj)
    plot_traj.plot_position_rmse(trajs, gt_step_traj)
    plot_traj.plot_total_position_rmse(trajs, gt_step_traj)
    plot_traj.plot_position_distance_error(trajs, gt_step_traj)
    # # Plot the training ouptut data
    # fig, axs = plt.subplots(2, 2)
    # axs = axs.flatten()

    # axs[0].plot(output_yawdiff)
    # for d in range(3):
    #     axs[d + 1].plot(output_pos[d, :])
    plt.show()