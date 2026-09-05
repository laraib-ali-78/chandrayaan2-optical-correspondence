"""Transformation Model Selection.
Implements the model hierarchy and automatic selection rules from Section 16:
Translation -> Similarity -> Affine -> Homography (default) -> Piecewise/DEM.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class TransformationModelSelector:
    """Selects and fits optimal transformation model based on scene extent and terrain relief."""

    SUPPORTED_MODELS = ["translation", "similarity", "affine", "homography", "dem_assisted"]

    @staticmethod
    def select_model(
        matches: List[Dict[str, Any]],
        swath_width_km: float = 3.0,
        dem_available: bool = False
    ) -> str:
        """
        Automatic model selection rule:
        - Homography is standard default for single OHRC / NAC frames (near-planar assumption reasonable).
        - If high relief with large terrain variation and DEM present -> dem_assisted.
        - If small local flat patch -> affine.
        """
        if dem_available and swath_width_km > 15.0:
            return "dem_assisted"
        elif len(matches) < 4:
            return "affine"
        else:
            return "homography"

    @staticmethod
    def fit_model(
        matches: List[Dict[str, Any]],
        model_type: str = "homography"
    ) -> Tuple[np.ndarray, float]:
        """Fits the specified transformation model and returns (matrix, rmse)."""
        src_pts = np.float32([m["src_pt"] for m in matches])
        ref_pts = np.float32([m["ref_pt"] for m in matches])

        if model_type == "affine" or len(matches) < 4:
            M, _ = cv2.estimateAffine2D(src_pts, ref_pts)
            if M is None:
                M = np.eye(2, 3, dtype=np.float32)
            # Convert 2x3 to 3x3
            H = np.eye(3, dtype=np.float64)
            H[:2, :3] = M
        else:  # homography
            H, _ = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 3.0)
            if H is None:
                H = np.eye(3, dtype=np.float64)

        # Compute RMSE
        residuals = []
        for m in matches:
            sx, sy = m["src_pt"]
            rx, ry = m["ref_pt"]
            p = H @ np.array([sx, sy, 1.0])
            if abs(p[2]) > 1e-6:
                px = p[0] / p[2]
                py = p[1] / p[2]
                residuals.append(np.sqrt((px - rx) ** 2 + (py - ry) ** 2))

        rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")
        return H, rmse

