import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing import Union, List, Optional, Tuple
from scipy.spatial.transform import Rotation, Slerp

from src.zupt_ins import orientation

_MM_TO_M = 0.001

@dataclass(frozen=True)
class TimeSeries :
    t: NDArray[np.floating] # (N,)   timestamps in seconds

    def __post_init__(self):
        if len(self.t.shape) != 1:
            raise ValueError(f"t must be 1D, got shape {self.t.shape}")
        
        if not np.all(np.diff(self.t)>0) :
            raise ValueError(f"t must be strictly increasing.")

    def __len__(self)->int:
        return self.t.shape[0]
    
    def __getitem__(self, index: Union[List[int], int, NDArray, slice]):
        return TimeSeries(
            t=self.t[index],
        )
    
    @staticmethod
    def truncate_to_overlap(*time_series: 'TimeSeries') -> Tuple:
        """
        Truncate multiple time series to their overlapping time interval.
        
        Args:
            *time_series: Variable number of TimeSeries objects to truncate.
        
        Returns:
            List of truncated TimeSeries objects with synchronized time intervals.
            Objects maintain their original class types (Trajectory, InertialData, etc.).
        
        Raises:
            ValueError: If no time series provided or if time intervals don't overlap.
        """        
        if len(time_series) < 2:
            raise ValueError("At least two TimeSeries must be provided")
        
        # Find overlapping time interval
        t_start = max(ts.t[0] for ts in time_series)
        t_end = min(ts.t[-1] for ts in time_series)
        
        if t_start >= t_end:
            raise ValueError(
                f"No overlapping time interval found. "
                f"Start times: {[ts.t[0] for ts in time_series]}, "
                f"End times: {[ts.t[-1] for ts in time_series]}"
            )
        
        # Truncate each time series to the overlapping interval
        truncated = []
        for ts in time_series:
            mask = (ts.t >= t_start) & (ts.t <= t_end)
            truncated.append(ts[mask])
        
        return tuple(truncated)
    
    @staticmethod
    def is_compatible(*time_series: 'TimeSeries') -> bool :
        if len(time_series) < 2:
            raise ValueError("At least two TimeSeries must be provided")
        
        
        base = time_series[0]
        for time_serie in time_series[1:] :
            if time_serie.t.shape != base.t.shape :
                return False
            if (np.max(np.abs(base.t - time_serie.t)) > 1e-9) :
                return False
        
        return True

    

@dataclass(frozen=True)
class Trajectory(TimeSeries):
    """
    Arguments
    ---------
    t : NDArray[np.floating], shape (N,)
    pos : NDArray[np.floating], shape (3,N)
    R_nb : NDArray[np.floating], shape (3,3,N)
    vel : Optional[np.floating], shape (3,N)
    """
    pos: NDArray[np.floating]      # (3, N)   position in metres
    R_nb: NDArray[np.floating]        # (3, 3, N) rotation matrices
    vel: Optional[NDArray[np.floating]] = None  # (3,N) velocity in m/s

    @property
    def euler_nb(self) -> NDArray[np.floating]:
        """Euler angle representation of orientations, shape (3,N)"""
        return np.asarray(Rotation.from_matrix(self.R_nb.transpose(2,0,1)).as_euler('xyz')).transpose(1,0)
    
    @property
    def quat_nb(self) -> NDArray[np.floating]:
        """Quaternion representation of orientations, shape (4,N)"""
        return np.asarray(Rotation.from_matrix(self.R_nb.transpose(2,0,1)).as_quat(scalar_first=True)).transpose(1,0)
    
    def __post_init__(self):
        super().__post_init__()
        N = len(self.t)
        if self.pos.shape != (3, N):
            raise ValueError(f"pos must have shape (3, {N}), got {self.pos.shape}")
        if self.R_nb.shape != (3, 3, N):
            raise ValueError(f"R must have shape (3, 3, {N}), got {self.R_nb.shape}")
        if self.vel is not None and self.vel.shape != (3, N):
            raise ValueError(f"vel must have shape (3, {N}), got {self.vel.shape}")

    def __getitem__(self, index: Union[List[int], int, NDArray, slice]) -> "Trajectory":
        return Trajectory(
            t=self.t[index],
            pos=self.pos[:,index],
            R_nb=self.R_nb[:,:,index]
        )

    @classmethod
    def from_csv(cls, path: Path) -> "Trajectory":
        """
        Load ground truth position and orientation from a CSV file.
        Expected columns:
            0     : timestamp (ms)
            2-4   : XYZ position (mm)
            5-13  : flattened row-major 3x3 rotation matrix
        """
        data = pl.read_csv(
            path, has_header=False
        ).filter(~pl.any_horizontal(pl.all().is_nan()))

        # Keep only rows where timestamp is strictly increasing
        data = data.filter(
            pl.col("column_1") > pl.col("column_1").shift(1).fill_null(float("-inf"))
        )

        pos_cols = [f"column_{i}" for i in range(2, 5)]
        rot_cols = [f"column_{i}" for i in range(5, 14)]

        pos_valid = sum(pl.col(c) ** 2 for c in pos_cols) != 0
        rot_valid = sum(pl.col(c) ** 2 for c in rot_cols) != 0

        data = data.filter(pos_valid & rot_valid)
        data = data.to_numpy()

        t = data[:, 0] / 1000.0
        pos = _MM_TO_M * data[:, 2:5].T            # (3, N)

        n = data.shape[0]
        R = (
            data[:, 5:14]
            .reshape(n, 3, 3)                      # MATLAB column-major reshape + transpose cancel
            .transpose(1, 2, 0)                    # (N, 3, 3) → (3, 3, N)
        )

        return cls(t=t, pos=pos, R_nb=R).clean()
    
    @classmethod
    def from_csv_int(cls, data_dir: Path, num: int) -> "Trajectory":
        return cls.from_csv(data_dir / f"{num}_Synchronized_Reference.csv")

    def clean(self) -> "Trajectory":
        """Remove entries with implausible Euler-angle jumps."""
        euler = orientation.matrix_to_euler(self.R_nb)
        roll, pitch = euler[0], euler[1]
        yaw = np.unwrap(euler[2])

        d_roll  = np.abs(np.diff(roll))
        d_pitch = np.abs(np.diff(pitch))
        d_yaw   = np.diff(yaw)

        bad = np.nonzero((d_roll > 0.3) | (d_pitch > 0.3) | (d_yaw > 0.3))[0]
        bad = np.concatenate([bad, bad + 1])

        keep = np.setdiff1d(np.arange(len(self.t)), bad)
        return self[keep]
    
    def temporal_alignment(self, inertial_t: NDArray) -> "Trajectory":
        """
        Aligns the instance trajectory to provided measurement timestamps via interpolation.

        Position is interpolated linearly. Rotations are interpolated using SLERP
        (Spherical Linear Interpolation) on SO(3), which guarantees valid rotation
        matrices and follows the shortest geodesic path between orientations.

        Parameters
        ----------
        inertial_t : np.ndarray, shape (N,)
            Timestamps of the inertial measurements in seconds.

        Returns
        -------
        Trajectory
            Instance trajectory resampled at the inertial timestamps,
            with the same time base as `inertial_t`.
        """
        t_gt  = self.t
        pos_gt = self.pos                        # (3, N)
        R_gt   = self.R_nb.transpose(2, 0, 1)       # (N, 3, 3)

        # --- zero-order-hold extension on the left ---
        if inertial_t[0] < t_gt[0]:
            t_gt   = np.concatenate([[inertial_t[0]], t_gt])
            pos_gt = np.hstack([pos_gt[:, :1], pos_gt])
            R_gt   = np.concatenate([R_gt[:1], R_gt], axis=0)

        # --- zero-order-hold extension on the right ---
        if inertial_t[-1] > t_gt[-1]:
            t_gt   = np.concatenate([t_gt, [inertial_t[-1]]])
            pos_gt = np.hstack([pos_gt, pos_gt[:, -1:]])
            R_gt   = np.concatenate([R_gt, R_gt[-1:]], axis=0)

        # Interpolate position
        pos = np.vstack([
            np.interp(inertial_t, t_gt, pos_gt[i])
            for i in range(3)
        ])

        # SLERP for rotations — (N, 3, 3) expected by Rotation
        slerp = Slerp(t_gt, Rotation.from_matrix(R_gt, assume_valid=False))
        R = slerp(inertial_t).as_matrix().transpose(1, 2, 0)  # back to (3, 3, N)

        return Trajectory(t=inertial_t, pos=pos, R_nb=R)

    def step_vectors_body(self, step_seg: List[int]) -> NDArray[np.floating]:
        """
        Compute body-frame step vectors from step segmentation indices.

        For each consecutive pair of step segment indices (k-1, k), the step
        vector is the displacement in the body frame at step k-1:
            step[:, k-1] = R_nb[:, :, seg[k-1]].T @ (pos[:, seg[k]] - pos[:, seg[k-1]])

        Args:
            step_seg: List of N_steps+1 indices marking step boundaries.

        Returns:
            steps: (3, N_steps) array of body-frame step vectors.
        """
        seg = np.asarray(step_seg)

        pos_start = self.pos[:, seg[:-1]]   # (3, N_steps)
        pos_end   = self.pos[:, seg[1:]]    # (3, N_steps)
        R_start   = self.R_nb[:, :, seg[:-1]]  # (3, 3, N_steps)

        displacements = pos_end - pos_start                         # (3, N_steps)
        steps = np.einsum('ijk,jk->ik', R_start.transpose(1, 0, 2), displacements)

        return steps

    def step_vectors_heading(self, step_seg: List[int]) -> NDArray[np.floating]:
        """
        Compute heading-frame step vectors from step segmentation indices.

        For each consecutive pair of step segment indices (k-1, k), the step
        vector is the displacement in the heading frame at step k-1:
            step[:, k-1] = R_nb[:, :, seg[k-1]].T @ (pos[:, seg[k]] - pos[:, seg[k-1]])

        Args:
            step_seg: List of N_steps+1 indices marking step boundaries.

        Returns:
            steps: (3, N_steps) array of heading-frame step vectors.
        """
        seg = np.asarray(step_seg)

        N_steps = seg[:-1].shape[0]

        pos_start = self.pos[:, seg[:-1]]   # (3, N_steps)
        pos_end   = self.pos[:, seg[1:]]    # (3, N_steps)
        R_start_nb   = self.R_nb[:, :, seg[:-1]]  # (3, 3, N_steps)

        # Compute euler angles
        euler_nb = Rotation.from_matrix(R_start_nb.transpose(2,0,1)).as_euler('xyz') # (N_steps,3)
        euler_nh = np.zeros((N_steps,1))
        euler_nh = euler_nb[:,2:3]

        # Compute new rotation matrices
        R_hn = Rotation.from_euler('z', euler_nh).as_matrix().transpose(2,1,0) # (N_steps,3,3)
        R_hn = np.asarray(R_hn)

        displacements = pos_end - pos_start                         # (3, N_steps)
        steps = np.einsum('ijk,jk->ik', R_hn, displacements)

        return steps

@dataclass(frozen=True)
class InertialData(TimeSeries):
    u: NDArray[np.floating]   # (6, N) stacked [accel; gyro]

    def __post_init__(self):
        super().__post_init__()
        N = len(self.t)
        if self.u.shape != (6, N):
            raise ValueError(f"u must have shape (6, {N}), got {self.u.shape}")

    def __getitem__(self, index: Union[int, List[int], NDArray, slice]) -> "InertialData":
        return InertialData(
            t=self.t[index],
            u=self.u[:,index]
        )

    @classmethod
    def from_csv(cls, path: Path ) -> "InertialData":
        """
        Load raw IMU data from a CSV file.

        Expected columns (1-indexed, after header):
            3     : timestamp
            4-6   : accelerometer (x, y, z)
            7-9   : gyroscope (x, y, z)
        """

        df = pl.read_csv(path, has_header=True)
        t     = df[:, 2].to_numpy()           # (N,)
        accel = df[:, 3:6].to_numpy().T       # (3, N)
        gyro  = df[:, 6:9].to_numpy().T       # (3, N)
        return cls(t=t, u=np.vstack([accel, gyro]))
    
    @classmethod
    def from_csv_int(cls, data_dir: Path, num: int) -> "InertialData":
        return cls.from_csv(data_dir / f"{num}_IMURaw.csv")



if __name__ == "__main__":

    imu = InertialData.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    gt = Trajectory.from_csv_int(PROJECT_ROOT / "data/angermann_high_precision", 15)
    
    # Example: Truncate to overlapping time interval
    imu_overlap, gt_overlap = TimeSeries.truncate_to_overlap(imu, gt)
    print("Original lengths:")
    print(f"  IMU: {len(imu)}")
    print(f"  GT:  {len(gt)}")
    print("\nOverlapping time interval:")
    print(f"  Start: {imu_overlap.t[0]:.3f}s")
    print(f"  End:   {imu_overlap.t[-1]:.3f}s")
    print("Truncated lengths:")
    print(f"  IMU: {len(imu_overlap)}")
    print(f"  GT:  {len(gt_overlap)}")
    
    print("\nIMU:")
    print(imu.u.shape)
    print(imu.u[:, 9999])

    print("\nGround truth:")
    print(gt.R_nb.shape)
    print(gt.t.shape)
    print(gt.pos.shape)
    idx = 9999
    print(gt.R_nb[:, :, idx])
    print("\nFirst element of rotation matrix :")
    print(f"{gt.R_nb[0,0,idx] = }")
    print(f"{gt.R_nb[2,2,idx] = }")

    print("\nFirst element of pos :")
    print(f"{gt.pos[0,idx] = }")
    print(f"{gt.pos[2,idx] = }")