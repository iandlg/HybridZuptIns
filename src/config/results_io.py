"""Utilities for saving and loading HSGP correction results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and types."""
    def default(self, obj): # type: ignore
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ResultsSaver:
    """Manages saving and loading of HSGP correction results with parameters."""
    
    def __init__(self, output_dir: str | Path):
        """
        Initialize results saver.
        
        Parameters
        ----------
        output_dir : str | Path
            Directory where results will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_run(
        self,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        metadata: Dict[str, Any] | None = None
    ) -> Path:
        """
        Save a run with parameters and results to a unique timestamped file.
        
        Parameters
        ----------
        parameters : Dict[str, Any]
            Dictionary of model parameters (m, ls, sigma_f, sigma_n, margin, etc.)
        results : Dict[str, Any]
            Dictionary of results (corrections, trajectories, metrics, etc.)
        metadata : Dict[str, Any], optional
            Additional metadata (e.g., data source, trial ID, frame)
        
        Returns
        -------
        Path
            Path to the saved file.
        
        Examples
        --------
        >>> saver = ResultsSaver("out/offline_correction/hsgp")
        >>> params = {"m": 500, "ls": 1.0, "sigma_f": 1.0, "sigma_n": 0.1, "margin": 3}
        >>> results = {"y_yaw": yaw_corrections, "y_pos": pos_corrections}
        >>> path = saver.save_run(params, results, metadata={"trial_id": 15})
        """
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"hsgp_run_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Prepare data structure
        data = {
            "timestamp": datetime.now().isoformat(),
            "parameters": parameters,
            "results": results,
            "metadata": metadata or {}
        }
        
        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
        
        return filepath
    
    @staticmethod
    def load_run(filepath: str | Path) -> Dict[str, Any]:
        """
        Load a saved run file.
        
        Parameters
        ----------
        filepath : str | Path
            Path to the saved run file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'parameters', 'results', and 'metadata' keys.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    
    def list_runs(self) -> list[Path]:
        """
        List all saved runs in the output directory, sorted by timestamp (newest first).
        
        Returns
        -------
        list[Path]
            List of paths to saved run files.
        """
        runs = sorted(
            self.output_dir.glob("hsgp_run_*.json"),
            reverse=True
        )
        return runs


def save_hsgp_run(
    output_dir: str | Path,
    parameters: Dict[str, Any],
    rmse_results: Dict[str, Any]
) -> Path:
    """
    Convenience function to save a complete HSGP correction run.
    
    Parameters
    ----------
    output_dir : str | Path
        Directory where results will be saved.
    parameters : Dict[str, Any]
        Model parameters (m, ls, sigma_f, sigma_n, margin, data_path, trial_id)
    rmse_results : Dict[str, Any]
        RMSE dictionary with trajectory names as keys and RMSE values
    trial_id : int, optional
        Trial ID for metadata.
    ref_frame : str, optional
        Reference frame for metadata.
    
    Returns
    -------
    Path
        Path to the saved file.
    
    Examples
    --------
    >>> save_hsgp_run(
    ...     "out/offline_correction/hsgp",
    ...     parameters={"m": 500, "ls": 1.0, "sigma_f": 1.0, "sigma_n": 0.1, "margin": 3, "data_path": "...", "trial_id": 15},
    ...     rmse_results={"model": 1.2, "model + HSGP": 0.8},
    ...     trajectories={"model + HSGP": hsgp_step_traj},
    ...     trial_id=15,
    ...     ref_frame="BOD"
    ... )
    """
    saver = ResultsSaver(output_dir)
    
    metadata = {}
    return saver.save_run(
        parameters=parameters,
        results={
            "rmse": rmse_results,
        },
        metadata=metadata
    )
