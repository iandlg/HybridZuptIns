import rootutils
PROJECT_ROOT = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Kernel
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict
import polars as pl

def compute_gp_corrections(
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        kernel: Kernel = (
            ConstantKernel() * RBF()
            + WhiteKernel()
        ),
        n_restarts_optimizer: int = 10
    ) -> Tuple[NDArray, NDArray]:
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
    y_testing_gp = np.zeros(n_samples)
    hyperparameters = np.zeros((10,4))
    D = 10

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
        model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=n_restarts_optimizer,
            alpha=0.0
        )
        model.fit(x_train, y_train)
        y_testing_gp[testing_ind] = model.predict(x_test)

        regression_rmse = np.sqrt(
            np.mean((y[testing_ind] - y_testing_gp[testing_ind]) ** 2)
        )

        if n_restarts_optimizer > 0 :
            batch_hyperparams = np.array([
                model.log_marginal_likelihood_value_,
                np.exp(model.kernel_.theta[0]/2),           # sigma_f
                np.exp(model.kernel_.theta[1]),             # length scale
                np.exp(model.kernel_.theta[2]/2)            # sigma_n 
            ])
            hyperparameters[i-1] = batch_hyperparams
                
    return y_testing_gp, hyperparameters


def hyperparameters_to_csv(
    hyperparams: dict,
    output_path: Path = PROJECT_ROOT / "out/hyperparameters/python/all_hyperparameters.csv"
) -> None:
    """
    Save hyperparameters from cross-validation folds to a CSV file.

    Parameters
    ----------
    hyperparams : dict
        Dictionary with keys ``"yaw"``, ``"pos_0"``, ``"pos_1"``, ``"pos_2"``.
        Each value is an ``(10, 5)`` array whose columns are
        ``[log_marginal_likelihood, regression_rmse, sigma_f, length_scale, sigma_n]``.
    output_path : str, optional
        Path to the output CSV file (relative to PROJECT_ROOT).
    """
    records = []
    for key, val in hyperparams.items():
        for fold_idx in range(val.shape[0]):
            row = val[fold_idx]  # [log_ml, sigma_f, length_scale, sigma_n]
            records.append({
                "output_type": key,
                "fold": fold_idx + 1,
                "log_marginal_likelihood": row[0],
                "sigma_f": row[1],
                "length_scale": row[2],
                "sigma_n": row[3],
            })
    
    df = pl.DataFrame(records)
    filename = PROJECT_ROOT / output_path
    if not filename.parent.exists():
        filename.parent.mkdir(exist_ok=True, parents=True)
    df.write_csv(filename)


def hyperparameters_from_csv(
    input_path: Path = PROJECT_ROOT / "out/hyperparameters/python/all_hyperparameters.csv"
) -> Dict[str, NDArray]:
    """
    Load hyperparameters from a CSV file and reconstruct the cross-validation structure.

    Parameters
    ----------
    input_path : Path, optional
        Path to the input CSV file.

    Returns
    -------
    hyperparams : Dict[str, NDArray]
        Dictionary with keys ``"yaw"``, ``"pos_0"``, ``"pos_1"``, ``"pos_2"``.
        Each value is an ``(10, 5)`` array whose columns are
        ``[log_marginal_likelihood, regression_rmse, sigma_f, length_scale, sigma_n]``.
    """
    filename = PROJECT_ROOT / input_path if isinstance(input_path, str) else input_path
    df = pl.read_csv(filename)
    
    hyperparams = {}
    output_types = df["output_type"].unique().to_list()
    
    for output_type in sorted(output_types):
        subset = df.filter(pl.col("output_type") == output_type).sort("fold")
        n_folds = subset.shape[0]
        
        data = np.zeros((n_folds, 5))
        data[:, 0] = subset["log_marginal_likelihood"].to_numpy()
        data[:, 1] = subset["sigma_f"].to_numpy()
        data[:, 2] = subset["length_scale"].to_numpy()
        data[:, 3] = subset["sigma_n"].to_numpy()
        
        hyperparams[output_type] = data
    
    return hyperparams


def set_fixed_kernel(hyperparameters: NDArray) -> Kernel :
    return (
            ConstantKernel(
                constant_value=hyperparameters[0]**2,
                constant_value_bounds="fixed"
            ) * RBF(
                length_scale=hyperparameters[1],
                length_scale_bounds="fixed"
            )
            + WhiteKernel(
                noise_level=hyperparameters[2]**2,
                noise_level_bounds="fixed"
            )
        )