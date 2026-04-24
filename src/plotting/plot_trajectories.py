import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import matplotlib.pyplot as plt
from matplotlib.pylab import Figure, Axes, Line2D
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

from src.zupt_ins.data_classes import Trajectory, InertialData

def plot_groundtruth_vs_inertial_traj(
        ins_traj: Trajectory,
        gt_traj: Trajectory,
):
    fig, ax = plt.subplots()

    ins_line, = ax.plot(ins_traj.pos[0, :-1], ins_traj.pos[1, :-1])
    ins_color = ins_line.get_color()

    ax.plot(gt_traj.pos[0, :-1], gt_traj.pos[1, :-1],
            color='black', linestyle='--', linewidth=1)
    
    marker_size = 30

    # Ground truth start/end markers.
    ax.scatter(*gt_traj.pos[:2, 0],  color='black', marker='o', s=marker_size, zorder=5)
    ax.scatter(*gt_traj.pos[:2, -2], color='black', marker='s', s=marker_size, zorder=5)

    # Inertial start/end markers.
    ax.scatter(*ins_traj.pos[:2, 0],  color=ins_color, marker='o', s=marker_size, zorder=5)
    ax.scatter(*ins_traj.pos[:2, -2], color=ins_color, marker='s', s=marker_size, zorder=5)

    ax.legend(
        handles=[
            Line2D([0], [0], color=ins_color, label='Inertial odometry'),
            Line2D([0], [0], color='black', linestyle='--', label='Ground truth'),
            Line2D([0], [0], color='grey', marker='o', linestyle='None', label='Start'),
            Line2D([0], [0], color='grey', marker='s', linestyle='None', label='End'),
        ]
    )

    ax.set_aspect('equal')
    ax.grid(visible=True)

if __name__ == "__main__":
    from src.zupt_ins.initialization import INSConfig
    from src.zupt_ins.data_classes import TimeSeries
    from src.zupt_ins.zupt_ins import smoothed_zupt_aided_ins
    from src.zupt_ins.data_preprocessing import temporal_alignment
    from src.zupt_ins.trajectory_transform import transform_trajectory

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


    ins_traj_aligned = transform_trajectory(ins_traj, gt_traj_aligned, zupt)

    plot_groundtruth_vs_inertial_traj(ins_traj_aligned[:1000], gt_traj_aligned[:1000])
    plt.show()