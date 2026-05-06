import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Optional

def plot_regression_results(
        y_yaw: NDArray,
        y_yaw_static: Optional[NDArray],
        y_yaw_gp: Optional[NDArray],
        y_pos: NDArray,
        y_pos_static: Optional[NDArray],
        y_pos_gp: Optional[NDArray]
):
    """
    Parameters
    ----------
    y_yaw : (N,) array
    y_yaw_static : (N,) array
    y_yaw_gp : (N,) array

    y_pos : (3, N) array
    y_pos_static : (3, N) array
    y_pos_gp : (3, N) array
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # --- YAW ---
    ax = axes[0]
    ax.plot(y_yaw, 'k', linewidth=1.2, label='True yaw error')
    if y_yaw_static is not None :
        ax.plot(y_yaw_static, '--b',
                linewidth=1.2,label=f'Static (RMSE = {np.sqrt(np.mean((y_yaw_static - y_yaw)**2)):.4f})')
        
    if y_yaw_gp is not None :    
        ax.plot(y_yaw_gp, '--r', linewidth=1.2,
                label=f'GP (RMSE = {np.sqrt(np.mean((y_yaw_gp - y_yaw)**2)):.4f})')

    ax.set_title('Yaw')
    ax.grid(True)
    ax.legend()

    # --- POSITION (X, Y, Z) ---
    labels = ['X position', 'Y position', 'Z position']

    for i in range(3):
        ax = axes[i + 1]

        ax.plot(y_pos[i], 'k', linewidth=1.2, label='True position')
        if y_pos_static is not None :
            rmse_static = np.sqrt(np.mean((y_pos_static[i] - y_pos[i])**2))
            ax.plot(y_pos_static[i], '--b', linewidth=1.2,
                        label=f'Static (RMSE = {rmse_static:.4f})')
        
        if y_pos_gp is not None :
                rmse_gp     = np.sqrt(np.mean((y_pos_gp[i]     - y_pos[i])**2))
                ax.plot(y_pos_gp[i], '--r', linewidth=1.2,
                        label=f'GP (RMSE = {rmse_gp:.4f})')

        ax.set_title(labels[i])
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
