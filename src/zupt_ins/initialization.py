from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Optional



@dataclass(frozen=True)
class INSConfig:
    """
    Configuration settings for the zero-velocity aided INS Kalman filter.

    Parameters
    ----------
    altitude : float
        Rough altitude (m).
    latitude : float
        Rough latitude (degrees).
    Ts : float
        Sampling period (s).
    init_heading : float
        Initial heading (rad).
    init_pos : tuple of float, length 3
        Initial position [x, y, z] (m).
    sigma_a : float
        Accelerometer noise standard deviation for detector (m/s^2).
    sigma_g : float
        Gyroscope noise standard deviation for detector (rad/s).
    window_size : int
        Zero-velocity detector window size (samples).
    gamma : float
        Zero-velocity detector threshold.
    sigma_acc : tuple of float, length 3
        Process noise std for accelerometer (m/s^2).
    sigma_gyro : tuple of float, length 3
        Process noise std for gyroscope (rad/s).
    sigma_vel : tuple of float, length 3
        Measurement noise std for velocity (m/s).
    sigma_initial_pos : tuple of float, length 3
        Initial position uncertainty std (m).
    sigma_initial_vel : tuple of float, length 3
        Initial velocity uncertainty std (m/s).
    sigma_initial_att : tuple of float, length 3
        Initial attitude uncertainty std [roll, pitch, heading] (rad).
    g : float
        Magnitude of local gravity vector (m/s^2). Computed from latitude
        and altitude if not provided.
    """

    # General
    altitude        : float         = 100.0
    latitude        : float         = 58.0
    Ts              : float         = 1 / 100
    init_heading    : float         = 0.0
    init_pos        : tuple         = (0.0, 0.0, 0.0)

    # Detector
    sigma_a         : float         = 0.01
    sigma_g         : float         = 0.2 * np.pi / 180
    window_size     : int           = 3
    gamma           : float         = 0.5e5

    # Filter — process noise
    sigma_acc       : tuple         = (1.3, 1.3, 1.3)
    sigma_gyro      : tuple         = (0.1 * np.pi / 180,) * 3

    # Filter — measurement noise
    sigma_vel       : tuple         = (0.1, 0.1, 0.1)

    # Filter — initial covariance
    sigma_initial_pos   : tuple     = (1e-5, 1e-5, 1e-5)
    sigma_initial_vel   : tuple     = (1e-5, 1e-5, 1e-5)
    sigma_initial_att   : tuple     = (100 * np.pi / 180,
                                       100 * np.pi / 180,
                                       0.1 * np.pi / 180)

    # Gravity — computed post-init if left as None
    g               : float         = None # type: ignore

    def __post_init__(self):
        if self.g is None:
            # Bypass frozen restriction for the derived field
            object.__setattr__(self, 'g', _gravity(self.latitude, self.altitude))

    # ------------------------------------------------------------------
    # Convenience accessors returning numpy arrays
    # ------------------------------------------------------------------
    @property
    def init_pos_array(self) -> NDArray[np.floating]:
        return np.array(self.init_pos, dtype=float)

    @property
    def sigma_acc_array(self) -> NDArray[np.floating]:
        return np.array(self.sigma_acc, dtype=float)

    @property
    def sigma_gyro_array(self) -> NDArray[np.floating]:
        return np.array(self.sigma_gyro, dtype=float)

    @property
    def sigma_vel_array(self) -> NDArray[np.floating]:
        return np.array(self.sigma_vel, dtype=float)

    @property
    def sigma_initial_pos_array(self) -> NDArray[np.floating]:
        return np.array(self.sigma_initial_pos, dtype=float)

    @property
    def sigma_initial_vel_array(self) -> NDArray[np.floating]:
        return np.array(self.sigma_initial_vel, dtype=float)

    @property
    def sigma_initial_att_array(self) -> NDArray[np.floating]:
        return np.array(self.sigma_initial_att, dtype=float)


def _gravity(latitude: float, altitude: float) -> float:
    """
    Compute local gravity magnitude using the WGS84 model.

    Parameters
    ----------
    latitude : float
        Latitude in degrees.
    altitude : float
        Altitude in metres.

    Returns
    -------
    g : float
        Local gravity magnitude (m/s^2).
    """
    lam   = np.pi / 180 * latitude
    gamma = 9.780327 * (1 + 0.0053024 * np.sin(lam)**2
                          - 0.0000058 * np.sin(2 * lam)**2)
    g     = (gamma
             - ((3.0877e-6) - (0.004e-6) * np.sin(lam)**2) * altitude
             + (0.072e-12) * altitude**2)
    return g