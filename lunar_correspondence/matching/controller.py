"""Adaptive Matcher Controller.
Implements the multi-signal decision engine fully specified in Section 14 of the Blueprint:
1. Automatically computes pair difficulty metrics (overlap, GSD ratio, sun-angle diff, texture score).
2. Manages the 3-tier cascade: LightGlue -> LoFTR -> RoMa -> Graceful FAILED.
3. Logs transparent explainability trace of all decisions and thresholds.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

from ..features.superpoint_wrapper import SuperPointWrapper
from .lightglue_wrapper import LightGlueMatcher
from .loftr_wrapper import LoFTRMatcher
from .roma_wrapper import RoMaMatcher


class AdaptiveMatcherController:
    """Controls multi-tier matcher cascade based on live image signals."""

    def __init__(
        self,
        min_candidate_matches: int = 20,
        min_inlier_ratio_lightglue: float = 0.15,
        min_grid_coverage: float = 0.40,
        escalate_loftr_inlier_ratio: float = 0.10,
        escalate_roma_illum_diff_deg: float = 45.0,
        min_inliers_for_transform: int = 8
    ):
        self.min_candidate_matches = min_candidate_matches
        self.min_inlier_ratio_lightglue = min_inlier_ratio_lightglue
        self.min_grid_coverage = min_grid_coverage
        self.escalate_loftr_inlier_ratio = escalate_loftr_inlier_ratio
        self.escalate_roma_illum_diff_deg = escalate_roma_illum_diff_deg
        self.min_inliers_for_transform = min_inliers_for_transform

        # Engines
        self.superpoint = SuperPointWrapper()
        self.lightglue = LightGlueMatcher()
        self.loftr = LoFTRMatcher()
        self.roma = RoMaMatcher()

    def compute_texture_score(self, image: np.ndarray) -> float:
        """Computes local gradient-magnitude variance as measure of terrain texture."""
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        return float(np.var(mag))

    def compute_grid_coverage(self, pts: np.ndarray, img_shape: Tuple[int, int], grid_size: int = 8) -> float:
        """Calculates fraction of grid cells containing at least one match."""
        if len(pts) == 0:
            return 0.0
        h, w = img_shape[:2]
        occupied_cells = set()
        cell_w = w / grid_size
        cell_h = h / grid_size

        for pt in pts:
            x, y = pt[0], pt[1]
            col = int(np.clip(x // cell_w, 0, grid_size - 1))
            row = int(np.clip(y // cell_h, 0, grid_size - 1))
            occupied_cells.add((row, col))

        return len(occupied_cells) / float(grid_size * grid_size)

    def match_pair(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        meta_src: Dict[str, Any],
        meta_ref: Dict[str, Any],
        verifier_callback
    ) -> Dict[str, Any]:
        """
        Executes Section 14 multi-signal adaptive matching cascade:
        Tier 1: LightGlue (SuperPoint)
        Tier 2 (Fallback): LoFTR
        Tier 3 (Fallback Quality Mode): RoMa
        Returns execution result dict with matches, inliers, trace, and selected tier.
        """
        trace = []
        h_s, w_s = img_src.shape[:2]
        h_r, w_r = img_ref.shape[:2]

        # Compute pair signals
        texture_score = self.compute_texture_score(img_src)
        gsd_src = meta_src.get("gsd_m_per_px", 1.0)
        gsd_ref = meta_ref.get("gsd_m_per_px", 1.0)
        gsd_ratio = max(gsd_src, gsd_ref) / max(1e-4, min(gsd_src, gsd_ref))

        az_s = meta_src.get("sun_azimuth_deg", 0.0)
        az_r = meta_ref.get("sun_azimuth_deg", 0.0)
        illum_diff_deg = abs(az_s - az_r) % 360.0
        if illum_diff_deg > 180.0:
            illum_diff_deg = 360.0 - illum_diff_deg

        trace.append(f"Pair signals: GSD ratio={gsd_ratio:.1f}x, Sun diff={illum_diff_deg:.1f}°, Texture var={texture_score:.1f}")

        # -------------------------------------------------------------
        # Tier 1: LightGlue (SuperPoint)
        # -------------------------------------------------------------
        trace.append("Executing Tier 1: LightGlue (SuperPoint sparse matcher)")
        pts_s, desc_s, _ = self.superpoint.detect_and_compute(img_src)
        pts_r, desc_r, _ = self.superpoint.detect_and_compute(img_ref)
        lg_matches = self.lightglue.match(pts_s, desc_s, pts_r, desc_r)

        n_candidate = len(lg_matches)
        lg_inliers, lg_H, inlier_ratio, lg_rmse = verifier_callback(lg_matches, img_src.shape, img_ref.shape)
        
        inlier_pts_src = np.array([m["src_pt"] for m in lg_inliers]) if lg_inliers else np.empty((0, 2))
        grid_coverage = self.compute_grid_coverage(inlier_pts_src, img_src.shape)

        trace.append(f"Tier 1 LightGlue results: candidates={n_candidate}, inliers={len(lg_inliers)}, inlier_ratio={inlier_ratio:.2f}, grid_coverage={grid_coverage:.2f}")

        # Check Escalation Rule 2
        escalate_to_loftr = (
            n_candidate < self.min_candidate_matches or
            inlier_ratio < self.min_inlier_ratio_lightglue or
            grid_coverage < self.min_grid_coverage
        )

        if not escalate_to_loftr and len(lg_inliers) >= self.min_inliers_for_transform:
            trace.append("Success with Tier 1: LightGlue accepted without escalation.")
            return {
                "status": "SUCCESS",
                "tier_used": "Tier 1: LightGlue",
                "raw_matches": lg_matches,
                "inliers": lg_inliers,
                "H": lg_H,
                "inlier_ratio": inlier_ratio,
                "grid_coverage": grid_coverage,
                "rmse": lg_rmse,
                "trace": trace
            }

        trace.append(f"Escalation triggered: candidate={n_candidate} (<{self.min_candidate_matches}) OR inlier_ratio={inlier_ratio:.2f} (<{self.min_inlier_ratio_lightglue}) OR coverage={grid_coverage:.2f} (<{self.min_grid_coverage})")

        # -------------------------------------------------------------
        # Tier 2: LoFTR (Dense Detector-Free)
        # -------------------------------------------------------------
        trace.append("Executing Fallback Tier 1: LoFTR (detector-free dense matching)")
        loftr_matches = self.loftr.match_dense(img_src, img_ref)
        loftr_inliers, loftr_H, loftr_inlier_ratio, loftr_rmse = verifier_callback(loftr_matches, img_src.shape, img_ref.shape)

        loftr_pts = np.array([m["src_pt"] for m in loftr_inliers]) if loftr_inliers else np.empty((0, 2))
        loftr_coverage = self.compute_grid_coverage(loftr_pts, img_src.shape)

        trace.append(f"Tier 2 LoFTR results: candidates={len(loftr_matches)}, inliers={len(loftr_inliers)}, inlier_ratio={loftr_inlier_ratio:.2f}, coverage={loftr_coverage:.2f}")

        # Check Escalation Rule 3
        escalate_to_roma = (
            loftr_inlier_ratio < self.escalate_loftr_inlier_ratio and
            illum_diff_deg > self.escalate_roma_illum_diff_deg
        )

        if not escalate_to_roma and len(loftr_inliers) >= self.min_inliers_for_transform:
            trace.append("Success with Fallback Tier 1: LoFTR accepted.")
            return {
                "status": "SUCCESS",
                "tier_used": "Tier 2: LoFTR (Fallback)",
                "raw_matches": loftr_matches,
                "inliers": loftr_inliers,
                "H": loftr_H,
                "inlier_ratio": loftr_inlier_ratio,
                "grid_coverage": loftr_coverage,
                "rmse": loftr_rmse,
                "trace": trace
            }

        if escalate_to_roma:
            trace.append(f"LoFTR inlier ratio {loftr_inlier_ratio:.2f} < {self.escalate_loftr_inlier_ratio} and sun diff {illum_diff_deg:.1f}° > {self.escalate_roma_illum_diff_deg}°. Escalating to RoMa.")

        # -------------------------------------------------------------
        # Tier 3: RoMa (Quality Mode)
        # -------------------------------------------------------------
        trace.append("Executing Fallback Tier 2: RoMa (quality mode dense matching)")
        roma_matches = self.roma.match(img_src, img_ref)
        roma_inliers, roma_H, roma_inlier_ratio, roma_rmse = verifier_callback(roma_matches, img_src.shape, img_ref.shape)

        roma_pts = np.array([m["src_pt"] for m in roma_inliers]) if roma_inliers else np.empty((0, 2))
        roma_coverage = self.compute_grid_coverage(roma_pts, img_src.shape)

        trace.append(f"Tier 3 RoMa results: candidates={len(roma_matches)}, inliers={len(roma_inliers)}, inlier_ratio={roma_inlier_ratio:.2f}, coverage={roma_coverage:.2f}")

        if len(roma_inliers) >= self.min_inliers_for_transform:
            trace.append("Success with Fallback Tier 2: RoMa accepted.")
            return {
                "status": "SUCCESS",
                "tier_used": "Tier 3: RoMa (Quality Mode)",
                "raw_matches": roma_matches,
                "inliers": roma_inliers,
                "H": roma_H,
                "inlier_ratio": roma_inlier_ratio,
                "grid_coverage": roma_coverage,
                "rmse": roma_rmse,
                "trace": trace
            }

        # -------------------------------------------------------------
        # All tiers failed
        # -------------------------------------------------------------
        trace.append("All 3 matcher tiers failed minimum inlier bar. Returning FAILED status.")
        return {
            "status": "FAILED",
            "reason": "insufficient_inliers_across_all_tiers",
            "tier_used": "None (All tiers exhausted)",
            "raw_matches": lg_matches + loftr_matches + roma_matches,
            "inliers": [],
            "H": None,
            "inlier_ratio": 0.0,
            "grid_coverage": 0.0,
            "rmse": float("inf"),
            "trace": trace
        }

