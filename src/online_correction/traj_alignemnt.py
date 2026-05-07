"""
Calibrated online INS initialization.

Pipeline:
  1. Run smoothed_zupt_aided_ins on inertial data up to `maximum_distance_m`
     (the INS stops at the last step index before that distance is exceeded).
  2. Rigidly align the short calibration trajectory to the ground truth using
     transform_position and transform_orientation — this gives us R_pos, t_pos,
     and R_ori.
  3. Rotate the *full* inertial data (accelerometer + gyroscope) by R_pos so the
     sensor frame matches the navigation frame assumed by the ground truth.
  4. Initialize a fresh smoothed_zupt_aided_ins from the aligned final states
     (position, velocity, attitude, quaternion) and run it over the remaining
     inertial data.
"""

import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
import matplotlib.pyplot as plt

from src.zupt_ins.initialization import INSConfig
from src.zupt_ins.zupt_ins import smoothed_zupt_aided_ins
from src.zupt_ins.data_classes import InertialData, Trajectory, TimeSeries
from src.zupt_ins.trajectory_transform import transform_position, transform_orientation
import src.zupt_ins.orientation as orientation
import src.plotting.plot_trajectories as plot_traj


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH  = PROJECT_ROOT / "data/angermann_high_precision"
TRIAL_ID   = 15

# How far the calibration segment should travel before stopping (metres).
CALIBRATION_DISTANCE_M = 3.0

# Full-run INS config (no distance limit).
SIM_CONFIG_FULL = INSConfig()

# Calibration-run INS config — stop after CALIBRATION_DISTANCE_M.
SIM_CONFIG_CALIB = INSConfig(maximum_distance_m=CALIBRATION_DISTANCE_M)


# ---------------------------------------------------------------------------
# Step 1 — load data and find overlapping window
# ---------------------------------------------------------------------------
inertial_full = InertialData.from_csv_int(DATA_PATH, TRIAL_ID)
gt_traj_full  = Trajectory.from_csv_int(DATA_PATH, TRIAL_ID)

inertial_trunc, gt_traj_trunc = TimeSeries.truncate_to_overlap(inertial_full, gt_traj_full)
gt_traj_aligned_full = gt_traj_trunc.temporal_alignment(inertial_trunc.t)

print(f"Total samples after truncation : {len(inertial_trunc)}")


# ---------------------------------------------------------------------------
# Step 2 — calibration run (short segment, stops at distance threshold)
# ---------------------------------------------------------------------------
print("\n--- Calibration INS run ---")
zupt_calib, ins_calib, segs_calib, x_end_calib, quat_end_calib = smoothed_zupt_aided_ins(
    inertial_trunc, SIM_CONFIG_CALIB
)

# The calibration run may be shorter than the full dataset — trim GT to match.
n_calib = len(ins_calib)
gt_calib = gt_traj_aligned_full[:n_calib]

print(f"Calibration trajectory length : {n_calib} samples")
print(f"Last step index               : {segs_calib[-1]}")


# ---------------------------------------------------------------------------
# Step 3 — rigid alignment of the calibration trajectory
# ---------------------------------------------------------------------------
print("\n--- Rigid alignment ---")

# Position alignment: finds rotation R_pos (yaw only + flip) and translation t_pos.
ins_calib_aligned, R_pos, t_pos = transform_position(
    ins_calib, gt_calib, zupt_calib, segs_calib
)

# Orientation alignment: finds residual rotation R_ori (roll/pitch/yaw trim).
ins_calib_aligned, R_ori = transform_orientation(
    ins_calib_aligned, gt_calib, zupt_calib, np.zeros(3), segs_calib
)

print(f"Position rotation R_pos:\n{R_pos}")
print(f"Position translation t_pos: {t_pos.ravel()}")
print(f"Orientation rotation R_ori:\n{R_ori}")

# Overall sensor-to-navigation rotation: first apply R_pos, then R_ori.
R_sensor_to_nav = R_ori.T @ R_pos   # applied left-to-right as: R_ori @ R_pos @ v


# ---------------------------------------------------------------------------
# Step 4 — transform the final states of the calibration run
# ---------------------------------------------------------------------------
# x_end_calib = [pos(3), vel(3), euler(3)]  at segs_calib[-1]
# quat_end_calib = quaternion at segs_calib[-1]

x_init = x_end_calib.copy()

# Rotate position and velocity into the aligned navigation frame.
x_init[0:3] = (R_pos @ x_init[0:3, None] + t_pos).ravel()
x_init[3:6] = (R_pos @ x_init[3:6, None]).ravel()

# Rotate attitude: apply R_ori post-multiply to the rotation matrix.
R_end = orientation.q2dcm(quat_end_calib)        # (3,3) rotation at last step
R_end_aligned = R_end @ R_ori                    # post-multiply

quat_init = orientation.dcm2q(R_end_aligned)
x_init[6:9] = orientation.matrix_to_euler(R_end_aligned)

print(f"\nInitial state for continuation run:")
print(f"  position  : {x_init[0:3]}")
print(f"  velocity  : {x_init[3:6]}")
print(f"  euler     : {np.rad2deg(x_init[6:9])} deg")


# ---------------------------------------------------------------------------
# Step 5 — rotate the remaining inertial data into the navigation frame
# ---------------------------------------------------------------------------
# Slice the inertial data starting one sample after the last calibration step.
continuation_start = segs_calib[-1]
inertial_cont = inertial_trunc[continuation_start:]
gt_cont = gt_traj_aligned_full[continuation_start:]

# Rotate accelerometer and gyroscope by R_pos so axes are consistent with the
# navigation frame established by the ground truth.
new_accel = R_pos @ inertial_cont.u[0:3, :]
new_gyro  = R_pos @ inertial_cont.u[3:6, :]
inertial_cont_rotated = InertialData(
    t=inertial_cont.t,
    u=np.vstack([new_accel, new_gyro])
)

print(f"\nContinuation segment starts at sample {continuation_start}")
print(f"Continuation segment length            : {len(inertial_cont_rotated)} samples")


# ---------------------------------------------------------------------------
# Step 6 — continuation INS run from aligned initial states
# ---------------------------------------------------------------------------
print("\n--- Continuation INS run ---")
zupt_cont, ins_cont, segs_cont, _, _ = smoothed_zupt_aided_ins(
    inertial_cont_rotated,
    SIM_CONFIG_FULL,
    # Pass the aligned final state as the initial condition.
    # smoothed_zupt_aided_ins calls initialize_nav internally for sample 0,
    # so we monkey-patch x[:,0] and quat[:,0] by injecting via a thin wrapper.
)

# NOTE: smoothed_zupt_aided_ins always re-initialises from the IMU data.
# To honour x_init / quat_init we need to call the lower-level routine.
# Re-run using the internal entry point.
from src.zupt_ins.zupt_ins import (
    init_filter, navigation_equations, state_matrix,
    compensate_internal_states, StepDetector
)
from src.zupt_ins import detector as det

u   = inertial_cont_rotated.u
Ts  = SIM_CONFIG_FULL.Ts
g   = SIM_CONFIG_FULL.g

zupt_c, _ = det.detector(u, SIM_CONFIG_FULL)
N = len(zupt_c)

Q_f, R_f, H_f = init_filter(SIM_CONFIG_FULL)
Id = np.eye(9)

x            = np.zeros((9, N))
quat         = np.zeros((4, N))
dx           = np.zeros((9, N))
dx_timeupd   = np.zeros((9, N))
dx_smooth    = np.zeros((9, N))
P            = np.zeros((9, 9, N))
P_timeupd    = np.zeros((9, 9, N))
P_smooth     = np.zeros((9, 9, N))
F_mat        = np.zeros((9, 9, N))

# Use aligned initial state.
x[:, 0]    = x_init
quat[:, 0] = quat_init

# Tight initial covariance — we trust the calibration alignment.
P[0:3, 0:3, 0] = np.diag([1e-4, 1e-4, 1e-4])
P[3:6, 3:6, 0] = np.diag([1e-4, 1e-4, 1e-4])
P[6:9, 6:9, 0] = np.diag(SIM_CONFIG_FULL.sigma_initial_att_array**2)

seg_start    = 1
seg_end      = N - 1
step_detector = StepDetector()
step_seg_cont: list[int] = []

while True:
    # Forward filter
    for n in range(seg_start, seg_end + 1):
        x[:, n], quat[:, n] = navigation_equations(
            x[:, n - 1], u[:, n], quat[:, n - 1], Ts, g
        )
        F_mat[:, :, n], G = state_matrix(quat[:, n], u[:, n], Ts)

        dx[:, n]   = F_mat[:, :, n] @ dx[:, n - 1]
        P[:, :, n] = F_mat[:, :, n] @ P[:, :, n - 1] @ F_mat[:, :, n].T + G @ Q_f @ G.T

        dx_timeupd[:, n]   = dx[:, n]
        P_timeupd[:, :, n] = P[:, :, n]

        if zupt_c[n]:
            K          = (P[:, :, n] @ H_f.T) @ np.linalg.inv(H_f @ P[:, :, n] @ H_f.T + R_f)
            dx[:, n]   = dx[:, n] - K @ (dx[3:6, n] - x[3:6, n])
            P[:, :, n] = (Id - K @ H_f) @ P[:, :, n]

        P[:, :, n] = (P[:, :, n] + P[:, :, n].T) / 2

        detected = step_detector(n, zupt_c[n])
        if detected is not None:
            step_seg_cont.append(detected)
            seg_end = n
            break

    # RTS smoother
    dx_smooth[:, seg_end]   = dx[:, seg_end]
    P_smooth[:, :, seg_end] = P[:, :, seg_end]

    for n in range(seg_end - 1, seg_start - 1, -1):
        A = P[:, :, n] @ F_mat[:, :, n].T @ np.linalg.inv(P_timeupd[:, :, n + 1])
        dx_smooth[:, n]   = dx[:, n] + A @ (dx_smooth[:, n + 1] - dx_timeupd[:, n + 1])
        P_smooth[:, :, n] = P[:, :, n] + A @ (P_smooth[:, :, n + 1] - P_timeupd[:, :, n + 1]) @ A.T
        P_smooth[:, :, n] = (P_smooth[:, :, n] + P_smooth[:, :, n].T) / 2

    x[:, seg_start:seg_end+1], quat[:, seg_start:seg_end+1] = compensate_internal_states(
        x[:, seg_start:seg_end+1],
        -dx_smooth[:, seg_start:seg_end+1],
        quat[:, seg_start:seg_end+1]
    )

    dx[:, seg_end]     = 0.0
    P[0:2, 8, seg_end] = 0.0
    P[8, 0:2, seg_end] = 0.0

    if seg_end != N - 1:
        seg_start = seg_end + 1
        seg_end   = N - 1
    else:
        break

ins_cont = Trajectory(
    t    = inertial_cont_rotated.t,
    pos  = x[0:3, :],
    R_nb = orientation.euler_to_matrix(x[6:9, :]),
    vel  = x[3:6, :]
)

print(f"Continuation trajectory length : {len(ins_cont)} samples")
print(f"Continuation step count        : {len(step_seg_cont)}")


# ---------------------------------------------------------------------------
# Step 7 — visualise
# ---------------------------------------------------------------------------
n_plot = min(len(ins_cont), len(gt_cont))

plot_traj.plot_groundtruth_vs_inertial_positions(
    {"Calibration (aligned)": ins_calib_aligned,},
    gt_calib,
)

plot_traj.plot_groundtruth_vs_inertial_positions(
    {"Continuation": ins_cont},
    gt_cont[:n_plot],
)

plot_traj.plot_groundtruth_vs_inertial_orientations(
    {"Continuation": ins_cont},
    gt_cont[:n_plot],
)

plot_traj.plot_position_rmse(
    {"Continuation": ins_cont},
    gt_cont[:n_plot],
)

plt.show()