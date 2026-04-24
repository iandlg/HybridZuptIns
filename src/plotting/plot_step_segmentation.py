import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import matplotlib.pyplot as plt
from typing import List, Union
import numpy as np
from numpy.typing import NDArray

from src.zupt_ins.data_classes import Trajectory, InertialData

def plot_inertialdata_and_stepsegm(
        inertial: InertialData,
        segs: List[int]
    ):
    fig, ax = plt.subplots()
    ax.grid(visible=True)
    ax.plot(inertial.t, (inertial.u[1:3, :] ** 2).sum(axis=0), linewidth=0.5, zorder=2)
    ax.scatter(inertial.t[segs], 100 * np.ones_like(segs), marker='x', c='r', s=5, zorder=1)


def plot_step_lengths(
        trajs: Union[List[Trajectory], Trajectory],
        gt_traj: Trajectory,
        segs: List[int],
):
    if isinstance(trajs, Trajectory):
        trajs = [trajs]

    gt_step_lengths = np.sqrt(
        np.sum(np.diff(gt_traj.pos[:, segs].T, axis=0) ** 2, axis=1)
    )

    fig, ax = plt.subplots()
    ax.plot(gt_traj.t[segs[:-1]], gt_step_lengths,
            color='black', linestyle='--', linewidth=1, label='Ground truth')

    for i, traj in enumerate(trajs):
        step_lengths = np.sqrt(
            np.sum(np.diff(traj.pos[:, segs].T, axis=0) ** 2, axis=1)
        )
        ax.plot(traj.t[segs[:-1]], step_lengths,
                label=getattr(traj, 'name', f'Trajectory {i + 1}'))

    ax.set_title('Length of steps')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Step length (m)')
    ax.grid(visible=True)
    ax.legend()

def plot_step_vectors(
        steps_ins: NDArray[np.floating],
        steps_gt: NDArray[np.floating],
):
    """
    Plots steps from ground truth and ins in the coordinate frame of the initial stance phase of the step

    Parameters
    --------
    steps_ins : NDArray[np.floating], shape (3,N)
        step from INS
    steps_gt : NDArray[np.floating], shape (3,N)
    """
    inds_plot = np.where(steps_gt[0, :] > -1.5)[0]

    fig, ax = plt.subplots()

    # Connecting lines between matched steps.
    for k in inds_plot:
        ax.plot(
            [steps_ins[0, k], steps_gt[0, k]],
            [steps_ins[1, k], steps_gt[1, k]],
            'k-', linewidth=0.8,
        )

    # Ground truth and inertial step markers.
    p1 = ax.scatter(steps_gt[0, inds_plot],  steps_gt[1, inds_plot],
                    color='black',   marker='x', linewidths=2,   s=40, label='Ground truth')
    p2 = ax.scatter(steps_ins[0, inds_plot], steps_ins[1, inds_plot],
                    color='magenta', marker='s', linewidths=1.5, s=40, label='Inertial odometry')

    # Axis cross.
    axis_range = np.arange(-2, 2.01, 0.01)
    ax.plot(axis_range, np.zeros_like(axis_range), 'k--', linewidth=0.8)
    ax.plot(np.zeros_like(axis_range), axis_range, 'k--', linewidth=0.8)

    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-1.5, 1.25)
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Position (m)')
    ax.set_aspect('equal')
    ax.grid(visible=True)
    ax.legend(handles=[p1, p2])

    plt.tight_layout()

if __name__ == "__main__":
    from src.zupt_ins.initialization import INSConfig
    from src.zupt_ins.zupt_ins import smoothed_zupt_aided_ins
    from src.zupt_ins.data_classes import TimeSeries
    from src.zupt_ins.trajectory_transform import transform_position, transform_orientation

    # Load data
    inertial = InertialData.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    gt_traj = Trajectory.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    simdata = INSConfig(segmentation_thrsld=0.03)

    # Data preprocessing to fit gt to imu data
    inertial_trunc, gt_traj_trunc = TimeSeries.truncate_to_overlap(inertial, gt_traj)
    gt_traj_aligned = gt_traj_trunc.temporal_alignment(inertial_trunc.t)

    # Compute trajectory from inertial data
    zupt, ins_traj, segs = smoothed_zupt_aided_ins(inertial_trunc, simdata)

    # Rigidly transform the positions and orientations of the computed trajectory
    ins_traj_aligned = transform_position(ins_traj, gt_traj_aligned, zupt)
    ins_traj_aligned = transform_orientation(ins_traj_aligned, gt_traj_aligned, zupt, np.zeros(3))

    # Compute local step vectors for ground truth and inertial trajectory
    steps_ins = ins_traj_aligned.step_vectors(segs)
    steps_gt = gt_traj_aligned.step_vectors(segs)

    # Plot 
    plot_inertialdata_and_stepsegm(inertial_trunc, segs)
    plot_step_lengths(ins_traj, gt_traj_aligned, segs)
    plot_step_vectors(steps_ins, steps_gt)
    plt.show()
