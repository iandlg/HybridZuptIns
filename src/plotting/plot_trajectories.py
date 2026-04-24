import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import matplotlib.pyplot as plt
from matplotlib.pylab import Figure, Axes, Line2D
from typing import Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray

from src.zupt_ins.data_classes import Trajectory, InertialData
import src.zupt_ins.orientation as orientation 

def plot_groundtruth_vs_inertial_positions(
        trajs: Union[List[Trajectory], Trajectory],
        gt_traj: Trajectory,
):
    fig, ax = plt.subplots()

    ax.plot(gt_traj.pos[0, :-1], gt_traj.pos[1, :-1],
            color='black', linestyle='--', linewidth=1)
    ax.scatter(*gt_traj.pos[:2, 0],  color='black', marker='o', s=30, zorder=5)
    ax.scatter(*gt_traj.pos[:2, -2], color='black', marker='s', s=30, zorder=5)

    if isinstance(trajs, Trajectory):
        trajs = [trajs]
    
    ins_handles = []
    for i, ins_traj in enumerate(trajs):
        ins_line, = ax.plot(ins_traj.pos[0, :-1], ins_traj.pos[1, :-1],
                            label=f'Inertial odometry {i + 1}')
        c = ins_line.get_color()
        ax.scatter(*ins_traj.pos[:2, 0],  color=c, marker='o', s=30, zorder=5)
        ax.scatter(*ins_traj.pos[:2, -2], color=c, marker='s', s=30, zorder=5)
        ins_handles.append(Line2D([0], [0], color=c, label=ins_line.get_label()))

    ax.legend(handles=[
        *ins_handles,
        Line2D([0], [0], color='black', linestyle='--', label='Ground truth'),
        Line2D([0], [0], color='grey', marker='o', linestyle='None', label='Start'),
        Line2D([0], [0], color='grey', marker='s', linestyle='None', label='End'),
    ])

    ax.set_aspect('equal')
    ax.grid(visible=True)
    plt.show()

def plot_groundtruth_vs_inertial_orientations(
        ins_traj: Trajectory,
        gt_traj: Trajectory,
):
    ins_euler = orientation.matrix_to_euler(ins_traj.R)  # (3, N)
    gt_euler = orientation.matrix_to_euler(gt_traj.R)    # (3, N)

    labels = ['Roll', 'Pitch', 'Yaw']
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    for i, (ax, label) in enumerate(zip(axs, labels)):
        ins_line, = ax.plot(ins_traj.t, np.rad2deg(ins_euler[i]), linewidth=0.8)
        ax.plot(gt_traj.t, np.rad2deg(gt_euler[i]),
                color='black', linestyle='--', linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Degrees')
        ax.grid(visible=True)

    axs[0].legend(
        handles=[
            Line2D([0], [0], color=ins_line.get_color(), label='Inertial odometry'), # type: ignore
            Line2D([0], [0], color='black', linestyle='--', label='Ground truth'),
        ]
    )

    fig.tight_layout()
    plt.show()

def plot_position_rmse(
        trajs: Union[List[Trajectory], Trajectory],
        gt_traj: Trajectory,
):
    if isinstance(trajs, Trajectory):
        trajs = [trajs]

    fig, ax = plt.subplots()

    for i, traj in enumerate(trajs):
        n = min(traj.pos.shape[1], gt_traj.pos.shape[1])
        rmse = np.sqrt(
            np.cumsum(np.sum((traj.pos[:2, :n] - gt_traj.pos[:2, :n]) ** 2, axis=0))
            / np.arange(1, n + 1)
        )
        ax.plot(traj.t[:n], rmse, label=getattr(traj, 'name', f'Trajectory {i + 1}'))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('Position RMSE')
    ax.legend()
    ax.grid(visible=True)
    


if __name__ == "__main__":
    from src.zupt_ins.initialization import INSConfig
    from src.zupt_ins.data_classes import TimeSeries
    from src.zupt_ins.zupt_ins import smoothed_zupt_aided_ins
    from src.zupt_ins.data_preprocessing import temporal_alignment
    from src.zupt_ins.trajectory_transform import transform_position, transform_orientation

    # Load data
    inertial = InertialData.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    gt_traj = Trajectory.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    simdata = INSConfig(segmentation_thrsld=0.03)

    # Data preprocessing to fit gt and imu data
    print("Original lengths:")
    print(f"  IMU: {len(inertial)}")
    print(f"  GT:  {len(gt_traj)}")
    interial_trunc, gt_traj_trunc = TimeSeries.truncate_to_overlap(inertial, gt_traj)
    print("\nOverlapping time interval:")
    print(f"  Start: {inertial.t[0]:.3f}s")
    print(f"  End:   {inertial.t[-1]:.3f}s")
    print("Truncated lengths:")
    print(f"  IMU: {len(interial_trunc)}")
    print(f"  GT:  {len(gt_traj_trunc)}")

    gt_traj_aligned = temporal_alignment(interial_trunc.t, gt_traj_trunc)

    # Compute trajectory from inertial data
    zupt, ins_traj, segs = smoothed_zupt_aided_ins(interial_trunc, simdata)


    ins_traj_aligned = transform_position(ins_traj, gt_traj_aligned, zupt)
    ins_traj_aligned = transform_orientation(ins_traj_aligned, gt_traj_aligned, zupt, np.zeros(3))

    plot_groundtruth_vs_inertial_positions(ins_traj_aligned[:1000], gt_traj_aligned[:1000])
    plot_groundtruth_vs_inertial_orientations(ins_traj_aligned[:], gt_traj_aligned[:])
    plot_position_rmse(ins_traj_aligned, gt_traj_aligned)
    plt.show()