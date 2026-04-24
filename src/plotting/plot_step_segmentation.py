import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import matplotlib.pyplot as plt
from typing import List, Union, Optional
import numpy as np
from numpy.typing import NDArray

from src.zupt_ins.data_classes import Trajectory, InertialData

def plot_inertialdata_and_stepsegm(
        inertial: InertialData,
        segs: List[int]
    ):
    fig, ax = plt.subplots()
    ax.grid(visible=True)
    ax.plot(inertial.t, (inertial.u[0:3, :] ** 2).sum(axis=0), linewidth=0.5, zorder=2)
    ax.scatter(inertial.t[segs], 100 * np.ones_like(segs), marker='x', c='r', s=5, zorder=1)


def plot_step_lengths(
        trajs: Union[List[Trajectory], Trajectory],
        gt_traj: Trajectory,
        segs: List[int],
): 
    """
    Plots 3D length between consecutive steps with respect to time
    """
    if isinstance(trajs, Trajectory):
        trajs = [trajs]
    
    dims = slice(0,3)
    print(f"{gt_traj.pos[dims, segs].T.shape = }")
    gt_step_lengths = np.sqrt(
        np.sum(np.diff(gt_traj.pos[dims, segs].T, axis=0) ** 2, axis=1)
    )

    fig, ax = plt.subplots()
    ax.plot(gt_traj.t[segs[:-1]], gt_step_lengths,
            color='black', linestyle='--', linewidth=1, label='Ground truth')

    for i, traj in enumerate(trajs):
        step_lengths = np.sqrt(
            np.sum(np.diff(traj.pos[dims, segs].T, axis=0) ** 2, axis=1)
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

def plot_step_vector_components(
        steps_list: Union[List[NDArray[np.floating]], NDArray[np.floating]],
        steps_gt: NDArray[np.floating],
        t:NDArray[np.floating],
        labels: Optional[List[str]] = None,
):
    """
    Plot step vector components (horizontal length, vertical, yaw angle)
    for ground truth and one or more inertial step vector arrays.

    Parameters
    ----------
    steps_list : NDArray shape (3, N) or list of such arrays
        Body-frame step vectors for one or more inertial trajectories.
    steps_gt : NDArray shape (3, N)
        Body-frame step vectors for ground truth.
    labels : list of str, optional
        Legend labels for each entry in steps_list.
    """
    if isinstance(steps_list, np.ndarray):
        steps_list = [steps_list]

    if labels is None:
        labels = [f'Trajectory {i + 1}' for i in range(len(steps_list))]

    def _horizontal(s): return np.sqrt(s[0] ** 2 + s[1] ** 2)
    def _vertical(s):   return s[2]
    def _yaw(s):        return np.rad2deg(np.arctan2(s[1], s[0]))

    components = [
        ('Horizontal length', 'm',   _horizontal),
        ('Vertical',          'm',   _vertical),
        ('Yaw angle',         'deg', _yaw),
    ]

    n_steps = steps_gt.shape[1]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (title, unit, fn) in zip(axs, components):
        ax.plot(t, fn(steps_gt), color='black', linestyle='--',
                linewidth=1, label='Ground truth')
        for steps, label in zip(steps_list, labels):
            ax.plot(t, fn(steps), label=label)
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(unit)
        ax.grid(visible=True)

    axs[0].legend()
    fig.tight_layout()
    plt.show()


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
    plot_step_lengths(ins_traj_aligned, gt_traj_aligned, segs)
    plot_step_vectors(steps_ins, steps_gt)

    # needs different navigation frame
    plot_step_vector_components(steps_ins, steps_gt, ins_traj_aligned.t[segs[:-1]])
    plt.show()
