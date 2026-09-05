"""MAGSAC++ Geometric Verification & Spatially Uniform Selection.
Implements Section 10 & 15 of the Blueprint:
1. MAGSAC++ (cv2.USAC_MAGSAC) with adaptive noise marginalization.
2. Spatially uniform grid/quadtree match selection (Blueprint Section 11/15).
3. Degenerate configuration mitigation (avoids single-crater clustering).
"""

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


class MAGSACVerifier:
    """Robust geometric verifier with MAGSAC++ and spatial uniformity filtering."""

    def __init__(
        self,
        threshold_px: float = 3.0,
        confidence: float = 0.999,
        max_iters: int = 5000,
        grid_rows: int = 8,
        grid_cols: int = 8,
        max_per_cell: int = 5
    ):
        self.threshold_px = threshold_px
        self.confidence = confidence
        self.max_iters = max_iters
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_per_cell = max_per_cell

    def verify_matches(
        self,
        matches: List[Dict[str, Any]],
        img_src_shape: Tuple[int, int],
        img_ref_shape: Tuple[int, int]
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], float, float]:
        """
        Runs MAGSAC++ homography estimation and computes inliers and RMSE.
        Returns:
            inliers: list of matching dicts passing geometric verification
            H: 3x3 homography matrix
            inlier_ratio: inliers / candidate_matches
            rmse: root mean square reprojection error over inliers
        """
        if len(matches) < 4:
            return [], None, 0.0, float("inf")

        src_pts = np.float32([m["src_pt"] for m in matches])
        ref_pts = np.float32([m["ref_pt"] for m in matches])

        # Try USAC_MAGSAC (Barath et al.) if available in OpenCV build
        method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC

        H, inlier_mask = cv2.findHomography(
            src_pts,
            ref_pts,
            method=method,
            ransacReprojThreshold=self.threshold_px,
            maxIters=self.max_iters,
            confidence=self.confidence
        )

        if H is None or inlier_mask is None:
            return [], None, 0.0, float("inf")

        inliers = []
        residuals = []
        for i, is_inlier in enumerate(inlier_mask.ravel()):
            if is_inlier:
                p_src = np.array([src_pts[i][0], src_pts[i][1], 1.0], dtype=np.float64)
                p_proj = H @ p_src
                if abs(p_proj[2]) > 1e-6:
                    p_proj_x = p_proj[0] / p_proj[2]
                    p_proj_y = p_proj[1] / p_proj[2]
                    err = np.sqrt((p_proj_x - ref_pts[i][0]) ** 2 + (p_proj_y - ref_pts[i][1]) ** 2)
                else:
                    err = 999.0

                m_copy = dict(matches[i])
                m_copy["residual_error_px"] = float(err)
                inliers.append(m_copy)
                residuals.append(err)

        inlier_ratio = len(inliers) / float(len(matches))
        rmse = float(np.sqrt(np.mean(np.square(residuals)))) if len(residuals) > 0 else float("inf")

        return inliers, H, inlier_ratio, rmse

    def apply_spatial_uniformity_grid(
        self,
        inliers: List[Dict[str, Any]],
        img_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Selects a spatially uniform subset of matches using an NxM grid
        (Section 11/15) to prevent matches clustering around a single crater rim.
        """
        if len(inliers) == 0:
            return []

        h, w = img_shape[:2]
        cell_w = w / self.grid_cols
        cell_h = h / self.grid_rows

        grid_buckets = {}
        for m in inliers:
            sx, sy = m["src_pt"]
            col = int(np.clip(sx // cell_w, 0, self.grid_cols - 1))
            row = int(np.clip(sy // cell_h, 0, self.grid_rows - 1))
            key = (row, col)
            if key not in grid_buckets:
                grid_buckets[key] = []
            grid_buckets[key].append(m)

        uniform_matches = []
        for key, bucket in grid_buckets.items():
            # Sort by lowest residual error and highest confidence
            sorted_bucket = sorted(
                bucket,
                key=lambda item: item.get("confidence", 1.0) / (item.get("residual_error_px", 1.0) + 0.1),
                reverse=True
            )
            uniform_matches.extend(sorted_bucket[:self.max_per_cell])

        return uniform_matches

