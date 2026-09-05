"""Confidence and Uncertainty Estimation.
Implements the three-tier confidence logic from Blueprint Section 18:
HIGH | MEDIUM | LOW | FAILED
"""

from typing import Dict, Any, Tuple, List


class ConfidenceEstimator:
    """Evaluates registration metrics to assign an auditable confidence level and reasons."""

    def __init__(
        self,
        t_high_inlier_ratio: float = 0.35,
        c_high_grid_coverage: float = 0.50,
        e_high_rmse: float = 2.0,
        t_med_inlier_ratio: float = 0.15,
        c_med_grid_coverage: float = 0.30,
        minimum_inliers_for_transform: int = 8
    ):
        self.t_high_inlier_ratio = t_high_inlier_ratio
        self.c_high_grid_coverage = c_high_grid_coverage
        self.e_high_rmse = e_high_rmse
        self.t_med_inlier_ratio = t_med_inlier_ratio
        self.c_med_grid_coverage = c_med_grid_coverage
        self.minimum_inliers_for_transform = minimum_inliers_for_transform

    def estimate_confidence(self, metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Assigns confidence level:
        Returns: (confidence_level: str, reasons: List[str])
        """
        inlier_ratio = metrics.get("inlier_ratio", 0.0)
        grid_coverage = metrics.get("grid_coverage", 0.0)
        rmse = metrics.get("rmse_px")
        n_inliers = metrics.get("n_inliers", 0)

        reasons = []

        if n_inliers < self.minimum_inliers_for_transform or rmse is None or rmse == float("inf"):
            reasons.append(f"Inliers {n_inliers} below minimum threshold {self.minimum_inliers_for_transform}")
            return "FAILED", reasons

        is_high = (
            inlier_ratio >= self.t_high_inlier_ratio and
            grid_coverage >= self.c_high_grid_coverage and
            rmse <= self.e_high_rmse
        )

        if is_high:
            reasons.append(f"Inlier ratio {inlier_ratio:.2f} >= {self.t_high_inlier_ratio}")
            reasons.append(f"Grid coverage {grid_coverage:.2f} >= {self.c_high_grid_coverage}")
            reasons.append(f"RMSE {rmse:.2f} px <= {self.e_high_rmse} px")
            return "HIGH", reasons

        is_med = (
            inlier_ratio >= self.t_med_inlier_ratio and
            grid_coverage >= self.c_med_grid_coverage
        )

        if is_med:
            reasons.append(f"Inlier ratio {inlier_ratio:.2f} >= {self.t_med_inlier_ratio}")
            reasons.append(f"Grid coverage {grid_coverage:.2f} >= {self.c_med_grid_coverage}")
            return "MEDIUM", reasons

        # Low confidence: transform estimated but flagged unreliable
        reasons.append(f"Marginal fit: inlier ratio {inlier_ratio:.2f} or coverage {grid_coverage:.2f} below medium bars")
        return "LOW", reasons

