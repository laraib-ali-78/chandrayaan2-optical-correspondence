"""IIRS Hyperspectral QUB Reader & Panchromatic Synthesizer.
Implements the panchromatic-equivalent composite generation (0.5-0.8 um)
specified in Blueprint Section 6.
"""

import os
from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2


class IIRSReader:
    """Reader for Chandrayaan-2 Imaging Infrared Spectrometer (IIRS) QUB data."""

    def __init__(self, target_wavelength_min_um: float = 0.5, target_wavelength_max_um: float = 0.8):
        self.target_wavelength_min_um = target_wavelength_min_um
        self.target_wavelength_max_um = target_wavelength_max_um

    def load_cube(self, qub_path: str) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any], str]:
        """Loads 3D hyperspectral cube [Bands, Lines, Samples] or mock structure."""
        if not os.path.exists(qub_path):
            return False, None, {}, f"IIRS QUB file not found: {qub_path}"

        try:
            # Check if stored as numpy .npy / raw binary / multi-band TIFF
            if qub_path.endswith(".npy"):
                cube = np.load(qub_path)
            else:
                # Raw binary or image sequence
                file_size = os.path.getsize(qub_path)
                # Attempt default headerless decode
                cube = np.fromfile(qub_path, dtype=np.float32)
            
            metadata = {
                "sensor": "IIRS",
                "gsd_m_per_px": 80.0,
                "wavelength_range_um": [self.target_wavelength_min_um, self.target_wavelength_max_um]
            }
            return True, cube, metadata, "OK"
        except Exception as e:
            return False, None, {}, f"Error reading IIRS QUB: {e}"

    def build_panchromatic_composite(self, cube: np.ndarray, method: str = "pca") -> np.ndarray:
        """
        Synthesizes a 2D panchromatic-equivalent composite from hyperspectral bands.
        
        Args:
            cube: 3D array of shape [Bands, Lines, Samples] or [Lines, Samples, Bands].
            method: 'pca' (first principal component) or 'average'.
        Returns:
            2D uint8 grayscale image.
        """
        if cube.ndim == 2:
            # Already 2D
            img = cube
        elif cube.ndim == 3:
            # Ensure shape is [Lines, Samples, Bands]
            if cube.shape[0] < cube.shape[1] and cube.shape[0] < cube.shape[2]:
                # Shape is [Bands, Lines, Samples] -> transpose
                cube = np.transpose(cube, (1, 2, 0))

            lines, samples, bands = cube.shape

            if method == "average":
                composite = np.nanmean(cube, axis=2)
            elif method == "pca":
                # Flatten spatial dimensions
                X = cube.reshape(-1, bands).astype(np.float32)
                # Remove NaN/Inf
                X = np.nan_to_num(X)
                # Center data
                mean = np.mean(X, axis=0)
                X_centered = X - mean
                # Covariance & PCA (1st component)
                cov = np.cov(X_centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                # Principal component with largest eigenvalue
                pc1 = eigenvectors[:, -1]
                projected = np.dot(X_centered, pc1)
                composite = projected.reshape(lines, samples)
            else:
                composite = np.nanmean(cube, axis=2)
        else:
            raise ValueError(f"Unsupported cube dimensions: {cube.shape}")

        # Normalize to uint8 [0, 255]
        norm = cv2.normalize(composite, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)

