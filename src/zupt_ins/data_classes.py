import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing import Union, List, Optional

from src.zupt_ins import orientation

_MM_TO_M = 0.001

@dataclass(frozen=True)
class TimeSeries :
    t: NDArray[np.floating] # (N,)   timestamps in seconds

    def __post_init__(self):
        if len(self.t.shape) != 1:
            raise ValueError(f"t must be 1D, got shape {self.t.shape}")

    def __len__(self)->int:
        return self.t.shape[0]
    

@dataclass(frozen=True)
class Trajectory(TimeSeries):
    pos: NDArray[np.floating]      # (3, N)   position in metres
    R: NDArray[np.floating]        # (3, 3, N) rotation matrices
    vel: Optional[NDArray[np.floating]] = None  # (3,N) velocity in m/s

    def __post_init__(self):
        super().__post_init__()
        N = len(self.t)
        if self.pos.shape != (3, N):
            raise ValueError(f"pos must have shape (3, {N}), got {self.pos.shape}")
        if self.R.shape != (3, 3, N):
            raise ValueError(f"R must have shape (3, 3, {N}), got {self.R.shape}")
        if self.vel is not None and self.vel.shape != (3, N):
            raise ValueError(f"vel must have shape (3, {N}), got {self.vel.shape}")

    def __getitem__(self, index: Union[List[int], int, NDArray]):
        return Trajectory(
            t=self.t[index],
            pos=self.pos[:,index],
            R=self.R[:,:,index]
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
        data = pl.read_csv(path, has_header=False).to_numpy()

        t = data[:, 0] / 1000.0                              # ms → s
        pos = _MM_TO_M * data[:, 2:5].T                      # (3, N)

        n = data.shape[0]
        R = (
            data[:, 5:14]
            .reshape(n, 3, 3)
            .transpose(0, 2, 1)   # row-major → column-major per slice
            .transpose(1, 2, 0)   # (N, 3, 3) → (3, 3, N)
        )
        np.nan_to_num(R, copy=False, nan=0.0)

        valid = (pos ** 2).sum(axis=0).astype(bool) & (R ** 2).sum(axis=(0, 1)).astype(bool)
        return cls(t=t[valid], pos=pos[:, valid], R=R[:, :, valid]).clean()
    
    @classmethod
    def from_csv_int(cls, data_dir: Path, num: int) -> "Trajectory":
        return cls.from_csv(data_dir / f"{num}_Synchronized_Reference.csv")

    def clean(self) -> "Trajectory":
        """Remove entries with implausible Euler-angle jumps."""
        euler = orientation.matrix_to_euler(self.R)
        roll, pitch = euler[0], euler[1]
        yaw = np.unwrap(euler[2])

        d_roll  = np.abs(np.diff(roll))
        d_pitch = np.abs(np.diff(pitch))
        d_yaw   = np.abs(np.diff(yaw))

        bad = np.nonzero((d_roll > 0.3) | (d_pitch > 0.3) | (d_yaw > 0.3))[0]
        bad = np.unique(np.concatenate([bad, bad + 1]))

        keep = np.setdiff1d(np.arange(len(self.t)), bad)
        return self[keep]
    

@dataclass(frozen=True)
class InertialData(TimeSeries):
    u: NDArray[np.floating]   # (6, N) stacked [accel; gyro]

    def __post_init__(self):
        super().__post_init__()
        N = len(self.t)
        if self.u.shape != (6, N):
            raise ValueError(f"u must have shape (6, {N}), got {self.u.shape}")

    def __getitem__(self, index: Union[int, List[int], NDArray])->'InertialData':
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
    print("IMU:")
    print(imu.u.shape)
    print(imu.u[:, 2000])

    print("\nGround truth:")
    print(gt.R.shape)
    print(gt.t.shape)
    print(gt.pos.shape)
    print(gt.R[:, :, 19999])