"""
Hyperparameter variability analysis for the GP correction model.

Add these two functions to src/offline_correction/exact_gp.py (or import them from here).
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import polars as pl

import src.offline_correction.batch_correction as correction
import src.offline_correction.gp as gp
from src.zupt_ins.data_classes import Trajectory, ReferenceFrame

def _save_fold_hyperparams(
        fold_idx: int,
        hyperparams: Dict[str, NDArray],
        rmse_per_fold: NDArray,
        output_file: Optional[Path]
    ) -> None:
    if output_file is None:
        return
    
    records = []
    for key in ["yaw", "pos_0", "pos_1", "pos_2"]:
        row = hyperparams[key][fold_idx]  # [log_ml, sigma_f, length_scale, sigma_n]
        records.append({
            "output_type": key,
            "fold":        fold_idx + 1,
            "log_marginal_likelihood":      row[0],
            "sigma_f":     row[1],
            "length_scale": row[2],
            "sigma_n":     row[3],
            "rmse":        rmse_per_fold[fold_idx],
        })
    
    output_df = pl.DataFrame(records)
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_csv(output_file)


def evaluate_hyperparameter_variability(
    ins_traj_aligned: Trajectory,
    gt_traj_aligned: Trajectory,
    segs: List[int],
    hyperparams: Dict[str, NDArray],
    ref_frame: ReferenceFrame = ReferenceFrame.BODY,
    output_filename: Optional[Path] = None,
    output_filename_non_outlier: Optional[Path] = None
) -> Tuple[NDArray, List[Trajectory]]:
    """
    For each of the 10 cross-validation hyperparameter sets, fix the GP kernel,
    recompute corrections and apply them, then record the 2-D position RMSE.

    Parameters
    ----------
    ins_traj_aligned : Trajectory
        INS trajectory already rigidly aligned to ground truth.
    gt_traj_aligned : Trajectory
        Ground truth trajectory sampled at the same timestamps.
    segs : List[int]
        Step-segment indices into the trajectories.
    hyperparams : dict
        Dictionary with keys ``"yaw"``, ``"pos_0"``, ``"pos_1"``, ``"pos_2"``.
        Each value is an ``(10, 4)`` array whose columns are
        ``[log_marginal_likelihood, sigma_f, length_scale, sigma_n]``,
        matching the output of ``compute_corrections``.
    ref_frame : ReferenceFrame
        Reference frame used to express step vectors (default: body frame).
    output_filename : Optional[Path]
        File path to save the fold with lowest RMSE.
    output_filename_non_outlier : Optional[Path]
        File path to save the fold with lowest RMSE that is not an outlier.

    Returns
    -------
    rmse_per_fold : NDArray, shape (10,)
        Horizontal 2-D position RMSE (metres) for each fold's kernel.
    corrected_trajs : list of Trajectory, length 10
        GP-corrected step-level trajectories, one per fold.
    """
    # ── training data (same for every fold) ──────────────────────────────────
    output_yawdiff, output_pos, input_feature = correction.compute_training_io(
        ins_traj_aligned, gt_traj_aligned, segs, ref_frame=ref_frame
    )

    n_folds = hyperparams["yaw"].shape[0]           # 10
    rmse_per_fold = np.empty(n_folds)
    corrected_trajs: List[Trajectory] = []

    gt_step_traj = gt_traj_aligned[segs]

    # Compute RMSE for all folds first
    for fold_idx in range(n_folds):
        # Build fixed kernels from this fold's hyperparameters
        # hyperparams[key][fold_idx] = [log_ml, sigma_f, length_scale, sigma_n]
        # set_fixed_kernel expects [sigma_f, length_scale, sigma_n]
        def _hp(key: str) -> NDArray:
            row = hyperparams[key][fold_idx]
            return row[1:4]          # drop log_marginal_likelihood

        kernel_yaw   = gp.set_fixed_kernel(_hp("yaw"))
        kernels_pos  = [gp.set_fixed_kernel(_hp(f"pos_{d}")) for d in range(3)]

        # GP predictions with the fixed kernel (no re-fitting)
        y_yaw_GP, _ = gp.compute_gp_corrections(
            input_feature, output_yawdiff, kernel=kernel_yaw, n_restarts_optimizer=0
        )

        y_pos_GP = np.empty(output_pos.shape)
        for d in range(3):
            y_pos_GP[d], _ = gp.compute_gp_corrections(
                input_feature, output_pos[d], kernel=kernels_pos[d], n_restarts_optimizer=0
            )

        # Apply corrections and build step-level corrected trajectory
        gp_traj = correction.apply_corrections(
            ins_traj_aligned, y_yaw_GP, y_pos_GP, segs, ref_frame=ref_frame
        )
        corrected_trajs.append(gp_traj)

        # 2-D horizontal RMSE against ground truth
        rmse_per_fold[fold_idx] = gp_traj.rmse(gt_step_traj)

    # Identify outliers using IQR method
    q1 = np.percentile(rmse_per_fold, 25)
    q3 = np.percentile(rmse_per_fold, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    is_outlier = (rmse_per_fold < lower_bound) | (rmse_per_fold > upper_bound)

    # Find best fold (lowest RMSE overall)
    best_fold_idx = int(np.argmin(rmse_per_fold))
    
    # Find best non-outlier fold (lowest RMSE among non-outliers)
    non_outlier_indices = np.where(~is_outlier)[0]
    if len(non_outlier_indices) > 0:
        best_non_outlier_idx = int(non_outlier_indices[np.argmin(rmse_per_fold[non_outlier_indices])])
    else:
        # If all folds are outliers, use the best fold overall
        best_non_outlier_idx = int(best_fold_idx)

    # Save hyperparameters for both folds
    _save_fold_hyperparams(best_fold_idx, hyperparams, rmse_per_fold, output_filename)
    _save_fold_hyperparams(best_non_outlier_idx, hyperparams, rmse_per_fold, output_filename_non_outlier)

    return rmse_per_fold, corrected_trajs

def plot_hyperparameter_rmse_variability(
    rmse_per_fold: NDArray,
    ins_rmse: float,
    static_rmse: float,
) -> None:
    """
    Visualise the spread of RMSE values across the 10 hyperparameter folds,
    together with the baseline (INS only) and static-mean-correction RMSE.

    Parameters
    ----------
    rmse_per_fold : NDArray, shape (10,)
        Per-fold RMSE from ``evaluate_hyperparameter_variability``.
    ins_rmse : float
        RMSE of the uncorrected aligned INS trajectory (baseline).
    static_rmse : float
        RMSE after applying a static (mean) correction.
    """
    n_folds = len(rmse_per_fold)
    fold_ids = np.arange(1, n_folds + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("GP Hyperparameter Variability — Position RMSE", fontsize=13)

    # ── Left: per-fold bar chart with baselines ───────────────────────────────
    ax = axs[0]
    bars = ax.bar(fold_ids, rmse_per_fold, color="steelblue", alpha=0.75,
                  edgecolor="white", linewidth=0.6, label="GP (fixed kernel per fold)")
    ax.axhline(ins_rmse,    color="black",   linestyle="--", linewidth=1.2, label="INS only")
    ax.axhline(static_rmse, color="tomato",  linestyle=":",  linewidth=1.2, label="Static correction")
    ax.axhline(rmse_per_fold.mean(), color="steelblue", linestyle="-",
               linewidth=1.0, alpha=0.6, label=f"GP mean ({rmse_per_fold.mean():.4f} m)")

    # Annotate each bar with its value
    for bar, val in zip(bars, rmse_per_fold):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="steelblue"
        )

    ax.set_xlabel("Hyperparameter fold")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("RMSE per fold")
    ax.set_xticks(fold_ids)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(visible=True, axis="y", which="both", alpha=0.4)
    ax.legend(fontsize=8)

    # ── Right: box plot + scatter overlay ────────────────────────────────────
    ax2 = axs[1]
    bp = ax2.boxplot(rmse_per_fold, vert=True, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2),
                     boxprops=dict(facecolor="steelblue", alpha=0.5),
                     whiskerprops=dict(linestyle="--"),
                     positions=[1], widths=0.4)

    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=n_folds)
    ax2.scatter(np.ones(n_folds) + jitter, rmse_per_fold,
                color="steelblue", s=40, zorder=5, alpha=0.85, label="Folds")

    ax2.axhline(ins_rmse,    color="black",  linestyle="--", linewidth=1.2, label="INS only")
    ax2.axhline(static_rmse, color="tomato", linestyle=":",  linewidth=1.2, label="Static correction")

    ax2.set_xticks([])
    ax2.set_ylabel("RMSE (m)")
    ax2.set_title(
        f"Distribution   μ={rmse_per_fold.mean():.4f} m   "
        f"σ={rmse_per_fold.std():.4f} m"
    )
    ax2.grid(visible=True, axis="y", alpha=0.4)
    ax2.legend(fontsize=8)

    fig.tight_layout()
