"""Evaluation Metrics for Lunar Image Correspondence.
Implements the exact formulas from Section 20 of the Blueprint.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class RegistrationMetrics:
    """Computes full quantitative metric suite defined in Section 20."""

    @staticmethod
    def compute_metrics(
        matches: List[Dict[str, Any]],
        inliers: List[Dict[str, Any]],
        H_estimated: Optional[np.ndarray],
        img_shape: tuple,
        H_ground_truth: Optional[np.ndarray] = None,
        gt_tie_points: Optional[List[tuple]] = None,
        grid_size: int = 8,
        rmse_acceptance_threshold: float = 2.5
    ) -> Dict[str, Any]:
        """
        Calculates all Section 20 metrics:
        - inlier_ratio
        - rmse, mean_error, median_error, max_error
        - registration_success (RMSE < threshold)
        - grid_coverage, match_distribution_score (coefficient of variation)
        - % sub-pixel error < 0.5px and < 1.0px against ground truth
        """
        n_candidates = len(matches)
        n_inliers = len(inliers)
        inlier_ratio = n_inliers / float(n_candidates) if n_candidates > 0 else 0.0

        residuals = []
        if H_estimated is not None and n_inliers > 0:
            for m in inliers:
                sx, sy = m.get("x", m.get("src_pt", (0, 0))[0]), m.get("y", m.get("src_pt", (0, 0))[1])
                rx, ry = m.get("x_ref", m.get("ref_pt", (0, 0))[0]), m.get("y_ref", m.get("ref_pt", (0, 0))[1])
                p = H_estimated @ np.array([sx, sy, 1.0])
                if abs(p[2]) > 1e-6:
                    px, py = p[0] / p[2], p[1] / p[2]
                    err = np.sqrt((px - rx) ** 2 + (py - ry) ** 2)
                    residuals.append(err)

        if residuals:
            residuals_arr = np.array(residuals)
            rmse = float(np.sqrt(np.mean(np.square(residuals_arr))))
            mean_error = float(np.mean(residuals_arr))
            median_error = float(np.median(residuals_arr))
            max_error = float(np.max(residuals_arr))
            success = bool(rmse <= rmse_acceptance_threshold)
        else:
            rmse = float("inf")
            mean_error = float("inf")
            median_error = float("inf")
            max_error = float("inf")
            success = False

        # Spatial distribution across grid
        h, w = img_shape[:2]
        cell_w, cell_h = w / grid_size, h / grid_size
        grid_counts = np.zeros((grid_size, grid_size), dtype=np.int32)

        for m in inliers:
            sx = m.get("x", m.get("src_pt", (0, 0))[0])
            sy = m.get("y", m.get("src_pt", (0, 0))[1])
            col = int(np.clip(sx // cell_w, 0, grid_size - 1))
            row = int(np.clip(sy // cell_h, 0, grid_size - 1))
            grid_counts[row, col] += 1

        occupied_cells = np.count_nonzero(grid_counts)
        grid_coverage = float(occupied_cells / float(grid_size * grid_size))

        # Coefficient of variation = std / mean
        mean_cnt = np.mean(grid_counts)
        match_distribution_score = float(np.std(grid_counts) / (mean_cnt + 1e-6))

        # Ground Truth Sub-Pixel Evaluation
        subpixel_under_05 = 0.0
        subpixel_under_10 = 0.0
        gt_rmse = None

        if H_ground_truth is not None and H_estimated is not None and gt_tie_points is not None and len(gt_tie_points) > 0:
            gt_errors = []
            for (pt_src, pt_ref_gt) in gt_tie_points:
                p = H_estimated @ np.array([pt_src[0], pt_src[1], 1.0])
                if abs(p[2]) > 1e-6:
                    px, py = p[0] / p[2], p[1] / p[2]
                    err = np.sqrt((px - pt_ref_gt[0]) ** 2 + (py - pt_ref_gt[1]) ** 2)
                    gt_errors.append(err)

            if gt_errors:
                gt_err_arr = np.array(gt_errors)
                gt_rmse = float(np.sqrt(np.mean(np.square(gt_err_arr))))
                subpixel_under_05 = float(np.mean(gt_err_arr < 0.5) * 100.0)
                subpixel_under_10 = float(np.mean(gt_err_arr < 1.0) * 100.0)

        return {
            "n_candidates": n_candidates,
            "n_inliers": n_inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "rmse_px": round(rmse, 3) if rmse != float("inf") else None,
            "mean_reprojection_error_px": round(mean_error, 3) if mean_error != float("inf") else None,
            "median_reprojection_error_px": round(median_error, 3) if median_error != float("inf") else None,
            "max_error_px": round(max_error, 3) if max_error != float("inf") else None,
            "registration_success": success,
            "grid_coverage": round(grid_coverage, 3),
            "match_distribution_score": round(match_distribution_score, 3),
            "gt_rmse_px": round(gt_rmse, 3) if gt_rmse is not None else None,
            "subpixel_pct_under_05px": round(subpixel_under_05, 1),
            "subpixel_pct_under_10px": round(subpixel_under_10, 1)
        }

