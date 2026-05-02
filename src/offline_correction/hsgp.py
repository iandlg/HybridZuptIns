import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Sequence

from sklearn.preprocessing import StandardScaler
    
def power_spectral_density(
        omega: NDArray,
        n_dims: int,
        ls: float | NDArray[np.floating],
        sigma_f: float = 1.0
    ) -> NDArray:
    """
    Power spectral density for the ExpQuad kernel.

    .. math::

        S(\\boldsymbol\\omega) =
            (\\sqrt(2 \\pi)^D \\prod_{i}^{D}\\ell_i
            \\exp\\left( -\\frac{1}{2} \\sum_{i}^{D}\\ell_i^2 \\omega_i^{2} \\right)

    Args:
        omega: array of shape (m_star, d), frequencies at which to evaluate the PSD.
        ls:    lengthscale(s), either a scalar or array of shape (d,).
        n_dims: number of input dimensions D.

    Returns:
        Array of shape (m_star,), one PSD value per basis function.
    """
    ls_arr = np.ones(n_dims) * ls          # (d,)
    c = np.power(np.sqrt(2.0 * np.pi), n_dims)
    exp = np.exp(-0.5 * np.dot(np.square(omega), np.square(ls_arr)))  # (m_star,)
    return sigma_f**2 * c * np.prod(ls_arr) * exp       # (m_star,)

def calc_eigenvalues(L: NDArray, m: Sequence[int]) -> NDArray:
    """Calculate eigenvalues of the Laplacian."""
    temp = [np.arange(1, 1 + m[d]) for d in range(len(m))]
    S = np.meshgrid(*temp)
    S_arr = np.vstack([s.flatten() for s in S]).T
    print(S_arr)
    return np.square((np.pi * S_arr) / (2 * L))

def calc_eigenvectors(Xs: NDArray, L: NDArray, eigvals: NDArray) -> NDArray:
    """Calculate eigenvectors of the Laplacian.
    These are used as basis vectors in the HSGP approximation.

    Parameters
    ----------
    Xs      : NDArray, shape (n_samples, d)
    L       : NDArray, shape (d,)
    eigvals : NDArray, shape (m_star, d)
    m       : Sequence[int], shape (d,)

    Returns
    -------
    phi : NDArray, shape (n_samples, m_star)
    """
    # (1, m_star, d) * (n_samples, 1, d) -> (n_samples, m_star, d)
    term1 = np.sqrt(eigvals)[None, :, :]          # (1, m_star, d)
    term2 = Xs[:, None, :] + L[None, None, :]     # (n_samples, 1, d) + (1, 1, d)
    c = 1.0 / np.sqrt(L)                          # (d,)

    phi = c * np.sin(term1 * term2)               # (n_samples, m_star, d)
    return np.prod(phi, axis=-1)                  # (n_samples, m_star)


def compute_hsgp_corrections(
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        m: Sequence[int] | int = 25,
        ls: float | NDArray[np.floating] = 1.0,
        sigma_f: float = 1.0,
        sigma_n: float = 1.0,
        L: float | NDArray[np.floating] = 1.5
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

    if isinstance(m, int) : 
        m = [m]*n_dim
    
    if isinstance(L, float):
        L = np.ones(n_dim, dtype=float) * L

    eigvals = calc_eigenvalues(L, m)  # type: ignore # (m_start, d)

    # Scale inputs to unit variance — critical for RBF length scale to be meaningful
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.T)  # (n_samples, n_features)

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
        psd = power_spectral_density(omega, n_dim, ls, sigma_f) 

        # A = phi.T @ phi + sigma_n**2 * np.diag(1 / psd)
        # L_chol = np.linalg.cholesky(A)                          # (m_star, m_star)
        # alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, phi.T @ y_train))

        # y_testing_GP[testing_ind] = phi_star @ alpha    # predictive mean

        Lambda = np.diag(psd)                                        # (m_star, m_star)
        K_nm   = phi @ Lambda @ phi.T                                # (n_train, n_train)
        A      = K_nm + sigma_n**2 * np.eye(len(y_train))           # (n_train, n_train)
        alpha  = np.linalg.solve(A, y_train)                         # (n_train,)
        y_testing_GP[testing_ind] = phi_star @ Lambda @ phi.T @ alpha                # (n_test,)
        # var = sigma_f**2 - np.sum(phi_star @ np.linalg.solve(L_chol, phi_star.T).T, axis=1)  # predictive variance                

    return y_testing_GP


if __name__ == "__main__" : 
    import src.zupt_ins.pipeline as pipeline
    import src.offline_correction.gp as gp
    import src.offline_correction.batch_correction as batch_correction
    from src.zupt_ins.data_classes import ReferenceFrame

    # Compute INS trajectory
    ins_traj_aligned, gt_traj_aligned, zupt, segs = pipeline.compute_aligned_ins_trajectory(
        data_path=PROJECT_ROOT / "data/angermann_high_precision",
        trial_id=15,
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
    m = [50,50,20]
    L = np.array([5,5,2])

    mean_step_size = np.mean(np.linalg.norm(input_feature, axis=0))
    max_distance = np.max(np.linalg.norm(input_feature, axis=0))

    y_yaw_hsgp = compute_hsgp_corrections(
        input_feature,
        output_yawdiff,
        m = m,
        ls = hyperparameters["yaw"][0,2],
        sigma_f= hyperparameters["yaw"][0,1],
        sigma_n= hyperparameters["yaw"][0,3],
        L=L
    )

    y_pos_hsgp = np.empty(output_pos.shape)
    for d in range(3):
        y_pos_hsgp[d, :] = compute_hsgp_corrections(
            input_feature, output_pos[d, :],
            m = m,
            ls = hyperparameters[f"pos_{d}"][0,2],
            sigma_f= hyperparameters[f"pos_{d}"][0,1],
            sigma_n= hyperparameters[f"pos_{d}"][0,3],
            L=L
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
    import src.plotting.plot_trajectories as plot
    import matplotlib.pyplot as plt
    plot.plot_groundtruth_vs_inertial_positions(trajs, gt_step_traj[:20])
    plot.plot_groundtruth_vs_inertial_orientations(trajs, gt_step_traj)
    plot.plot_position_rmse(trajs, gt_step_traj)
    plot.plot_total_position_rmse(trajs, gt_step_traj)
    plot.plot_position_distance_error(trajs, gt_step_traj)
    plt.show()
