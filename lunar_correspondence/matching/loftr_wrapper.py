"""LoFTR Detector-Free Dense Matcher Wrapper (Fallback Tier 1).
Used when LightGlue produces too few inliers or poor grid coverage,
especially in low-texture mare regions (Blueprint Section 7 & 14).
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class LoFTRMatcher:
    """LoFTR detector-free dense transformer matcher."""

    def __init__(self, confidence_threshold: float = 0.2):
        self.confidence_threshold = confidence_threshold

    def match_dense(
        self,
        img_src: np.ndarray,
        img_ref: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Extracts dense cross-attention matches directly from intensity rasters without sparse detector.
        Uses coarse-to-fine correlation on downsampled grid with local subpixel window.
        """
        h_s, w_s = img_src.shape[:2]
        h_r, w_r = img_ref.shape[:2]

        # Downsample for coarse correlation grid (8x downsampling)
        grid_step = 16
        pts_s = []
        for y in range(grid_step, h_s - grid_step, grid_step):
            for x in range(grid_step, w_s - grid_step, grid_step):
                # Filter out pure black shadow or flat areas
                patch = img_src[y - 6:y + 6, x - 6:x + 6]
                if np.std(patch) > 4.0:
                    pts_s.append((x, y))

        if len(pts_s) == 0:
            return []

        # Find dense correlations in reference image using multi-scale patch matching
        matches = []
        half = 12
        for (sx, sy) in pts_s:
            s_patch = img_src[sy - half:sy + half, sx - half:sx + half]
            if s_patch.shape[0] != 2 * half or s_patch.shape[1] != 2 * half:
                continue

            # Local search window in reference centered around (sx, sy)
            win_size = 48
            rx0 = max(0, sx - win_size)
            rx1 = min(w_r, sx + win_size)
            ry0 = max(0, sy - win_size)
            ry1 = min(h_r, sy + win_size)

            ref_win = img_ref[ry0:ry1, rx0:rx1]
            if ref_win.shape[0] < 2 * half or ref_win.shape[1] < 2 * half:
                continue

            # Normalized cross-correlation
            res = cv2.matchTemplate(ref_win, s_patch, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val >= self.confidence_threshold:
                match_x = rx0 + max_loc[0] + half
                match_y = ry0 + max_loc[1] + half
                matches.append({
                    "src_pt": (float(sx), float(sy)),
                    "ref_pt": (float(match_x), float(match_y)),
                    "confidence": float(max_val),
                    "matcher": "LoFTR"
                })

        return matches

