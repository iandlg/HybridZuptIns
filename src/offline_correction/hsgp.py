import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Sequence

from sklearn.preprocessing import StandardScaler
from config.results_io import save_hsgp_run

type ArrayLike = NDArray | Sequence[float | int]

def power_spectral_density(
        omega: NDArray,
        ls: float | NDArray,
        n_dims: int, 
        sigma_f : float
    ) -> NDArray:
    """
    Power spectral density for the Squated Exponential (SE) kernel.

    .. math::

        S(\\boldsymbol\\omega) =
            \\sigma_f^2(\\sqrt(2 \\pi)^D \\prod_{i}^{D}\\ell_i
            \\exp\\left( -\\frac{1}{2} \\sum_{i}^{D}\\ell_i^2 \\omega_i^{2} \\right)

    Parameters
    ----------
        omega: NDArray (m_star, d),
            Frequencies at which to evaluate the PSD. Frequencies are per dimension.
        ls: float | NDArray,
            Length scale either a scalar or array of shape (d,).
        n_dims: int,
            Number of input dimensions d.
        sigma_f : float,
            Standard deviation of the squared exponential kernel

    Returns
    --------
        Array of shape (m_star,), one PSD value per basis function.
    """
    ls_arr = np.ones(n_dims) * ls          # (d,)
    c = np.power(np.sqrt(2.0 * np.pi), n_dims)
    exp = np.exp(-0.5 * np.dot(np.square(omega), np.square(ls_arr)))  # (m_star,)
    return sigma_f**2 * c * np.prod(ls_arr) * exp       # (m_star,)

def calc_eigenvalues(L: ArrayLike, m: int, d: int) -> NDArray:
    """
    Calculate eigenvalues of the Laplacian on [-L1,L1] x ... x [-Ld,Ld]
    with Dirichlet boundary conditions, returning the m smallest.

    Parameters
    ----------
    L : NDArray, shape (d,)
        Domain half-widths per dimension.
    m : int
        Number of eigenfunctions to return.
    d : int
        Number of input dimensions.

    Returns
    -------
    selected_per_dim_eigenvalues : NDArray, shape (m,d)
        The m smallest eigenvalues, sorted ascending.
    """
    L = np.asarray(L, dtype=float)
    
    # Number of indices per dimension, scaled by relative domain size
    N_per_dim = np.ceil(m ** (1 / d) * L / L.min()).astype(int)

    # Build full multi-index grid (Cartesian product of per-dim indices)
    temp = [np.arange(1, 1 + N_per_dim[dim]) for dim in range(d)]
    grids = np.meshgrid(*temp, indexing='ij')
    NN = np.vstack([g.ravel() for g in grids]).T      # (N_total, d)

    # Compute all eigenvalues
    per_dim_eigvals = np.square((np.pi * NN) / (2 * L)) # (N_total, d)
    all_eigenvalues = np.sum(per_dim_eigvals, axis=1)

    # Sort and keep the m smallest
    sort_idx = np.argsort(all_eigenvalues)[:m]
    selected_per_dim_eigenvalues = per_dim_eigvals[sort_idx]

    return selected_per_dim_eigenvalues

def calc_eigenvectors(Xs: NDArray, L: ArrayLike, per_dim_eigvals: NDArray) -> NDArray:
    """Calculate eigenvectors of the Laplacian.
    These are used as basis vectors in the HSGP approximation.

    Parameters
    ----------
    Xs              : NDArray, shape (n_samples, d)
    L               : NDArray, shape (d,)
    per_dim_eigvals : NDArray, shape (m, d)
        Per dimension eigenvalues, with the smallest sum

    Returns
    -------
    phi : NDArray, shape (n_samples, m)
    """
    L = np.asarray(L, dtype=float)
    
    # (1, m, d) * (n_samples, 1, d) -> (n_samples, m, d)
    term1 = np.sqrt(per_dim_eigvals)[None, :, :]          # (1, m, d)
    term2 = Xs[:, None, :] + L[None, None, :]     # (n_samples, 1, d) + (1, 1, d)
    c = 1.0 / np.sqrt(L)                          # (d,)

    phi = c * np.sin(term1 * term2)               # (n_samples, m, d)
    return np.prod(phi, axis=-1)                  # (n_samples, m)


def compute_hsgp_corrections(
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        m: int = 25,
        ls: float | NDArray = 1.0,
        sigma_f: float = 1.0,
        sigma_n: float = 1.0,
        margin: float = 1.8
    ) -> NDArray:
    """
    Estimate output corrections using leave-one-fold-out Gaussian Process regression.

    Performs 10-fold cross-validation, fitting a GP with an RBF + white noise
    kernel on each fold. Also computes a static (mean) correction as a baseline.

    Parameters
    -------
    x : NDArray[np.floating], shape (n_features, n_samples)
        Input features, one column per sample.
    y : NDArray[np.floating], shape (n_samples,)
        Scalar target values to regress.

    Returns
    -------
    y_testing_GP : NDArray[np.floating], shape (n_samples,)
        GP-predicted corrections assembled across all folds.
    y_testing_static : NDArray[np.floating], shape (n_samples,)
        Static correction, equal to the global mean of y for all samples.
    """
    n_samples = x.shape[1]
    n_dim = x.shape[0]
    y_testing_GP = np.zeros(n_samples)
    D = 10
    
    # Scale inputs to unit variance — critical for RBF length scale to be meaningful
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.T)  # (n_samples, n_features)

    # L should cover the scaled domain with some margin
    margin = 1.5
    L = margin * np.abs(x_scaled).max(axis=0)   # (d,) — per-dimension, post-scaling
    eigvals = calc_eigenvalues(L, m, n_dim)  # type: ignore # (m_start, d)

    for i in range(1, D + 1):
        # --- indices --------------------------------------------------------
        test_start = int(np.floor(n_samples / D * (i - 1)))
        test_end   = int(np.floor(n_samples / D * i))

        training_ind = list(range(0, test_start)) + list(range(test_end, n_samples))
        testing_ind  = list(range(test_start, test_end))

        # --- data -----------------------------------------------------------
        x_train = x_scaled[training_ind,:]   # (n_train, n_features)
        y_train = y[training_ind]        # (n_train, )
        x_test  = x_scaled[testing_ind, :]    # (n_test,  n_features)

        # --- fit + predict --------------------------------------------------
        phi = calc_eigenvectors(x_train, L, eigvals)  # (n_samples, m_star)
        phi_star = calc_eigenvectors(x_test, L, eigvals)
        omega = np.sqrt(eigvals)  # (m_star, d)
        psd = power_spectral_density(omega, ls, n_dim, sigma_f)

        Lambda_inv = np.diag(1/psd)

        # A = σ_n² Λ^{-1} + ΦᵀΦ
        A = Lambda_inv * sigma_n**2 + (phi.T @ phi)  # (m, m)

        # Cholesky factorisation: A = L Lᵀ
        L_chol = np.linalg.cholesky(A)          # (m, m), lower triangular

        # Solve A @ alpha = Φᵀ y  via two triangular solves
        # L v = Φᵀ y,  then Lᵀ alpha = v
        rhs   = phi.T @ y_train                          # (m,)
        v     = np.linalg.solve(L_chol, rhs)             # forward  substitution
        alpha = np.linalg.solve(L_chol.T, v)             # backward substitution

        y_testing_GP[testing_ind] = phi_star @ alpha     # (N_test,)

    return y_testing_GP


if __name__ == "__main__" : 
    import src.zupt_ins.pipeline as pipeline
    import src.offline_correction.gp as gp
    import src.offline_correction.batch_correction as batch_correction
    from src.zupt_ins.data_classes import ReferenceFrame

    # Save parameters and results
    data_path = PROJECT_ROOT / "data/angermann_high_precision"
    trial_id = 15

    # Compute INS trajectory
    ins_traj_aligned, gt_traj_aligned, zupt, segs = pipeline.compute_aligned_ins_trajectory(
        data_path=data_path,
        trial_id=trial_id,
    )

    FRAME = ReferenceFrame.BOD

    # Compute inputs and outputs for regression
    output_yawdiff, output_pos, input_feature = batch_correction.compute_training_io(
        ins_traj_aligned, gt_traj_aligned, segs, ref_frame=FRAME)
    
    # Load hyperparameters from variability results
    hyperparameters = gp.hyperparameters_from_csv(
        PROJECT_ROOT / "out/hyperparameters/python/hparam_variability_results.csv"
    )
    
    # Apply static correction
    y_yaw_static = batch_correction.compute_static_correctons(input_feature, output_yawdiff)
    y_pos_static = np.empty(output_pos.shape)
    for d in range(3):
        y_pos_static[d, :] = batch_correction.compute_static_correctons(input_feature, output_pos[d, :])
    static_step_traj = batch_correction.apply_corrections(ins_traj_aligned, y_yaw_static, y_pos_static, segs, FRAME)

    # Apply exact GP correction
    y_yaw_gp, _ = gp.compute_gp_corrections(
        input_feature, output_yawdiff,
        kernel=gp.set_fixed_kernel(
            hyperparameters["yaw"][0,1:4]
        ),
        n_restarts_optimizer=0
    )

    y_pos_gp = np.empty(output_pos.shape)
    for d in range(3):
        y_pos_gp[d], _ = gp.compute_gp_corrections(
            input_feature, output_pos[d],
            kernel=gp.set_fixed_kernel(
                hyperparameters[f"pos_{d}"][0,1:4]
            ),
            n_restarts_optimizer=0
        )
    gp_step_traj = batch_correction.apply_corrections(ins_traj_aligned, y_yaw_gp, y_pos_gp, segs, FRAME)

    # Apply HSGP correction
    m = 220
    margin = 2

    mean_step_size = np.mean(np.linalg.norm(input_feature, axis=0))
    max_distance = np.max(np.linalg.norm(input_feature, axis=0))

    y_yaw_hsgp = compute_hsgp_corrections(
        input_feature,
        output_yawdiff,
        m = m,
        ls = hyperparameters["yaw"][0,2],
        sigma_f= hyperparameters["yaw"][0,1],
        sigma_n= hyperparameters["yaw"][0,3],
        margin=margin
    )

    y_pos_hsgp = np.empty(output_pos.shape)
    for d in range(3):
        y_pos_hsgp[d, :] = compute_hsgp_corrections(
            input_feature, output_pos[d, :],
            m = m,
            ls = hyperparameters[f"pos_{d}"][0,2],
            sigma_f= hyperparameters[f"pos_{d}"][0,1],
            sigma_n= hyperparameters[f"pos_{d}"][0,3],
            margin=margin
        )

    hsgp_step_traj = batch_correction.apply_corrections(ins_traj_aligned, y_yaw_hsgp, y_pos_hsgp, segs, ref_frame=FRAME)
   
    isn_step_traj = ins_traj_aligned[segs]
    gt_step_traj = gt_traj_aligned[segs]
    trajs = {
        "model" : isn_step_traj,
        "model + static" : static_step_traj,
        "model + GP" : gp_step_traj,
        "model + HSGP" : hsgp_step_traj
        # "model + GP + dt" : GP_step_traj_aug
    }

    hyp_dict = {}
    for dim_key in ["yaw", "pos_0", "pos_1", "pos_2"]:
        hyp_dict[dim_key] = {}
        for idx, hyp_key in enumerate(["sigma_f", "ls", "sigma_n"]):
            hyp_dict[dim_key][hyp_key] = hyperparameters[dim_key][0,idx+1]


    parameters = {
        "data_path": str(data_path),
        "trial_id": trial_id,
        "local_reference_frame": str(FRAME),
        "m": m,
        "margin": margin,
        "hyperparameters" : hyp_dict,
    }
    
    # Compute RMSE for each trajectory
    rmse_results = {}
    for traj_name, traj in trajs.items():
        rmse_results[traj_name] = traj.rmse(gt_step_traj)
    
    filepath = save_hsgp_run(
        output_dir=PROJECT_ROOT / "out/offline_correction/hsgp",
        parameters=parameters,
        rmse_results=rmse_results,
    )
    print(f"Results saved to: {filepath.relative_to(PROJECT_ROOT)}")
    
    # Plot results
    import src.plotting.plot_trajectories as plot
    import matplotlib.pyplot as plt
    plot.plot_groundtruth_vs_inertial_positions(trajs, gt_step_traj[:20])
    plot.plot_groundtruth_vs_inertial_orientations(trajs, gt_step_traj)
    plot.plot_position_rmse(trajs, gt_step_traj)
    plot.plot_total_position_rmse(trajs, gt_step_traj)
    plot.plot_position_distance_error(trajs, gt_step_traj)
    plt.show()
