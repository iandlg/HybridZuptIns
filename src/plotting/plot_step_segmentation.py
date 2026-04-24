import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import matplotlib.pyplot as plt
from matplotlib.pylab import Figure, Axes
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray

from src.zupt_ins.data_classes import Trajectory, InertialData

def plot_step_segmentation(inertial: InertialData, segs: List[int]):
    fig, ax = plt.subplots()
    ax.grid(visible=True)
    ax.plot(inertial.t, (inertial.u[1:3, :] ** 2).sum(axis=0), linewidth=0.5, zorder=2)
    ax.scatter(inertial.t[segs], 100 * np.ones_like(segs), marker='x', c='r', s=5, zorder=1)


def plot_step_lengths(
        gt_traj: Trajectory,
        ins_traj: Trajectory,
        segs: List[int],
):
    gt_step_lengths = np.sqrt(
        np.sum(np.diff(gt_traj.pos[:, segs].T, axis=0) ** 2, axis=1)
    )
    inertial_step_lengths = np.sqrt(
        np.sum(np.diff(ins_traj.pos[:, segs].T, axis=0) ** 2, axis=1)
    )

    fig, ax = plt.subplots()
    ax.plot(gt_traj.t[segs[:-1]], gt_step_lengths, label='ground truth')
    ax.plot(ins_traj.t[segs[:-1]], inertial_step_lengths, label='inertial odometry')
    ax.set_title('Length of steps')
    ax.grid(visible=True)
    ax.legend()

if __name__ == "__main__":
    from src.zupt_ins.initialization import INSConfig
    from src.zupt_ins.zupt_ins import smoothed_zupt_aided_ins
    from src.zupt_ins.data_preprocessing import temporal_alignment

    # Load data
    inertial = InertialData.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    gt_traj = Trajectory.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    simdata = INSConfig(segmentation_thrsld=0.03)

    # Align gt to inertial data
    gt_traj = temporal_alignment(inertial.t, gt_traj)
    print(len(gt_traj))
    
    # Compute trajectory from inertial data
    zupt, ins_traj, segs = smoothed_zupt_aided_ins(inertial, simdata)

    # Plot 
    plot_step_segmentation(inertial, segs)

    plot_step_lengths(gt_traj, ins_traj, segs)
    plt.show()
