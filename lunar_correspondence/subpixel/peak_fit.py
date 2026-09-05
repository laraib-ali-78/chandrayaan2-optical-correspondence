"""Parabolic Peak-Fit Sub-Pixel Refinement.
Fallback sub-pixel refinement method on the local NCC surface (Section 17).
"""

from typing import Tuple
import cv2
import numpy as np


class ParabolicPeakFit:
    """Computes sub-pixel displacement by fitting a 2D quadratic paraboloid to the correlation peak."""

    @staticmethod
    def refine(
        patch_src: np.ndarray,
        patch_ref: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Refines integer match to sub-pixel precision.
        Returns: (dx, dy, peak_val)
        """
        # Cross correlation
        res = cv2.matchTemplate(patch_ref, patch_src, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        px, py = max_loc
        h, w = res.shape

        # If peak is at boundary, cannot fit paraboloid
        if px <= 0 or px >= w - 1 or py <= 0 or py >= h - 1:
            return 0.0, 0.0, float(max_val)

        # 1D parabolic fit along x
        # peak offset: delta = (f(x-1) - f(x+1)) / (2 * (f(x-1) - 2*f(x) + f(x+1)))
        f_left = res[py, px - 1]
        f_center = res[py, px]
        f_right = res[py, px + 1]

        denom_x = 2.0 * (f_left - 2.0 * f_center + f_right)
        dx = (f_left - f_right) / (denom_x + 1e-7) if abs(denom_x) > 1e-6 else 0.0
        dx = float(np.clip(dx, -0.9, 0.9))

        # 1D parabolic fit along y
        f_top = res[py - 1, px]
        f_bottom = res[py + 1, px]

        denom_y = 2.0 * (f_top - 2.0 * f_center + f_bottom)
        dy = (f_top - f_bottom) / (denom_y + 1e-7) if abs(denom_y) > 1e-6 else 0.0
        dy = float(np.clip(dy, -0.9, 0.9))

        return dx, dy, float(f_center)

