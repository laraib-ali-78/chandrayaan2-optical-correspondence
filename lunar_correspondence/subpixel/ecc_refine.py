"""ECC (Enhanced Correlation Coefficient) Sub-Pixel Refinement.
Primary sub-pixel alignment engine (Blueprint Section 17) with Parabolic Peak-Fit fallback.
Produces the exact output schema from Section 4.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from .peak_fit import ParabolicPeakFit


class ECCSubPixelRefiner:
    """Refines coarse pixel correspondences to sub-pixel accuracy."""

    def __init__(
        self,
        patch_size: int = 48,
        max_iterations: int = 40,
        epsilon: float = 1e-4
    ):
        self.patch_size = patch_size
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
        self.peak_fitter = ParabolicPeakFit()

    def refine_match(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        src_pt: Tuple[float, float],
        ref_pt: Tuple[float, float],
        coarse_confidence: float = 1.0,
        matcher_source: str = "LightGlue"
    ) -> Dict[str, Any]:
        """
        Refines a single point correspondence using ECC affine patch alignment.
        Falls back to parabolic peak-fit if ECC fails to converge.
        Returns match dict adhering to Section 4 Output Contract.
        """
        sx, sy = src_pt
        rx, ry = ref_pt
        half = self.patch_size // 2

        h_s, w_s = img_src.shape[:2]
        h_r, w_r = img_ref.shape[:2]

        isx, isy = int(round(sx)), int(round(sy))
        irx, iry = int(round(rx)), int(round(ry))

        # Check patch bounds
        if (isx - half < 0 or isx + half >= w_s or isy - half < 0 or isy + half >= h_s or
            irx - half < 0 or irx + half >= w_r or iry - half < 0 or iry + half >= h_r):
            # Patch clipped by edge, keep coarse coordinates
            return {
                "x": float(sx),
                "y": float(sy),
                "x_ref": float(rx),
                "y_ref": float(ry),
                "subpixel_dx": 0.0,
                "subpixel_dy": 0.0,
                "confidence": float(coarse_confidence),
                "residual_error_px": 0.0,
                "matcher_source": matcher_source,
                "refinement_status": "skipped_boundary"
            }

        p_src = img_src[isy - half:isy + half, isx - half:isx + half].astype(np.float32)
        p_ref = img_ref[iry - half:iry + half, irx - half:irx + half].astype(np.float32)

        # Normalize patches
        p_src = cv2.normalize(p_src, None, 0, 255, cv2.NORM_MINMAX)
        p_ref = cv2.normalize(p_ref, None, 0, 255, cv2.NORM_MINMAX)

        subpixel_dx = 0.0
        subpixel_dy = 0.0
        refined_conf = coarse_confidence
        status = "ecc"

        # Try ECC with Translation or Euclidean model
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix = cv2.findTransformECC(
                p_ref,
                p_src,
                warp_matrix,
                motionType=cv2.MOTION_TRANSLATION,
                criteria=self.criteria
            )
            subpixel_dx = float(warp_matrix[0, 2])
            subpixel_dy = float(warp_matrix[1, 2])

            # Bound sub-pixel shift to within +/- 2.5 px
            if abs(subpixel_dx) > 2.5 or abs(subpixel_dy) > 2.5:
                raise cv2.error("ECC shift exploded")

        except cv2.error:
            # Fallback to Parabolic Peak-Fit
            dx, dy, peak_corr = self.peak_fitter.refine(p_src, p_ref)
            subpixel_dx = dx
            subpixel_dy = dy
            refined_conf = max(0.1, peak_corr)
            status = "peak_fit_fallback"

        refined_x_ref = rx + subpixel_dx
        refined_y_ref = ry + subpixel_dy

        return {
            "x": float(sx),
            "y": float(sy),
            "x_ref": float(refined_x_ref),
            "y_ref": float(refined_y_ref),
            "subpixel_dx": float(subpixel_dx),
            "subpixel_dy": float(subpixel_dy),
            "confidence": float(refined_conf),
            "residual_error_px": float(np.sqrt(subpixel_dx ** 2 + subpixel_dy ** 2)),
            "matcher_source": matcher_source,
            "refinement_status": status
        }

    def refine_all_matches(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray,
        inliers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Refines a list of verified inlier correspondences."""
        refined_matches = []
        for m in inliers:
            rec = self.refine_match(
                img_src=img_src,
                img_ref=img_ref,
                src_pt=m["src_pt"],
                ref_pt=m["ref_pt"],
                coarse_confidence=m.get("confidence", 1.0),
                matcher_source=m.get("matcher", "LightGlue")
            )
            refined_matches.append(rec)
        return refined_matches

