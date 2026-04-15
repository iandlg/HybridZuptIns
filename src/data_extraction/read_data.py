import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing import Union, List

from src.orientation import orientation

_MM_TO_M = 1e-3 # 0.001 


@dataclass(frozen=True)
class GroundTruthData:
    t: NDArray        # (N,)     timestamps in seconds
    pos: NDArray      # (3, N)   position in metres
    R: NDArray        # (3, 3, N) rotation matrices

    def __getitem__(self, index: Union[List[int], int, NDArray]):
        return GroundTruthData(
            t=self.t[index],
            pos=self.pos[:,index],
            R=self.R[:,:,index]
        )

    @classmethod
    def from_csv(cls, path: Path) -> "GroundTruthData":
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

    def clean(self) -> "GroundTruthData":
        """Remove entries with implausible Euler-angle jumps."""
        euler = orientation.rotation_stack_to_euler(self.R)
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
class IMUData:
    t: NDArray   # (N,)   timestamps
    u: NDArray   # (6, N) stacked [accel; gyro]

    def __getitem__(self, index: Union[int, List[int], NDArray])->'IMUData':
        return IMUData(
            t=self.t[index],
            u=self.u[:,index]
        )

    @classmethod
    def from_csv(cls, path: Path) -> "IMUData":
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


@dataclass(frozen=True)
class Dataset:
    imu: IMUData
    gt: GroundTruthData

    @classmethod
    def load(cls, data_dir: Path, num: int) -> "Dataset":
        """Load a paired IMU + ground-truth dataset by sequence number."""
        return cls(
            imu=IMUData.from_csv(data_dir / f"{num}_IMURaw.csv"),
            gt=GroundTruthData.from_csv(data_dir / f"{num}_Synchronized_Reference.csv"),
        )


if __name__ == "__main__":
    ds = Dataset.load(PROJECT_ROOT / "data/angermann_high_precision", 15)

    print("IMU:")
    print(ds.imu.u.shape)
    print(ds.imu.u[:, 2000])

    print("\nGround truth:")
    print(ds.gt.R.shape)
    print(ds.gt.t.shape)
    print(ds.gt.pos.shape)
    print(ds.gt.R[:, :, 19999])