"""Physical GSD-Aligned Multi-Scale Pyramid.
Resamples images to physically matched pixel scales (e.g., 20m, 5m, 1m, 0.3m/px)
rather than arbitrary dyadic scales (Blueprint Section 13).
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class GSDAlignedPyramid:
    """Builds physical resolution aligned pyramid levels for cross-sensor matching."""

    DEFAULT_LEVELS_M = [20.0, 5.0, 1.0, 0.3]  # meters per pixel

    def __init__(self, target_gsd_levels: List[float] = None):
        self.target_gsd_levels = sorted(target_gsd_levels or self.DEFAULT_LEVELS_M, reverse=True)

    def build_pyramid(self, image: np.ndarray, native_gsd_m: float) -> Dict[float, np.ndarray]:
        """
        Generates pyramid levels where pixel resolution matches the target physical GSD.
        Images are only downsampled, never naively upsampled past native resolution.
        """
        pyramid = {}
        h, w = image.shape[:2]

        for target_gsd in self.target_gsd_levels:
            if target_gsd >= native_gsd_m * 0.9:  # Target is coarser or equal
                scale_factor = native_gsd_m / target_gsd
                new_w = max(32, int(round(w * scale_factor)))
                new_h = max(32, int(round(h * scale_factor)))
                
                if abs(scale_factor - 1.0) < 0.05:
                    resampled = image.copy()
                else:
                    # Antialiased area interpolation for downsampling
                    resampled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

                pyramid[target_gsd] = resampled

        # Always include native resolution
        if native_gsd_m not in pyramid:
            pyramid[native_gsd_m] = image.copy()

        return pyramid

    def get_shared_levels(
        self,
        pyr_src: Dict[float, np.ndarray],
        pyr_ref: Dict[float, np.ndarray]
    ) -> List[float]:
        """Finds common physical GSD levels available in both image pyramids (sorted coarse to fine)."""
        src_levels = set(pyr_src.keys())
        ref_levels = set(pyr_ref.keys())
        shared = sorted(list(src_levels.intersection(ref_levels)), reverse=True)
        return shared

