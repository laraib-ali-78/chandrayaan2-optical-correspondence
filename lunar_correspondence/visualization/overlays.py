"""Visualization Tools for Lunar Image Correspondence.
Implements match overlays, inlier/outlier verification plots, and registration blends.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class Visualizer:
    """Generates visual diagnostics for match verification and registration."""

    @staticmethod
    def draw_matches_side_by_side(
        img_src: np.ndarray,
        img_ref: np.ndarray,
        matches: List[Dict[str, Any]],
        max_draw: int = 150
    ) -> np.ndarray:
        """Draws correspondence lines between source (left) and reference (right)."""
        h_s, w_s = img_src.shape[:2]
        h_r, w_r = img_ref.shape[:2]

        h = max(h_s, h_r)
        w = w_s + w_r

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        # Convert grayscale to BGR
        src_bgr = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR) if img_src.ndim == 2 else img_src
        ref_bgr = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if img_ref.ndim == 2 else img_ref

        canvas[:h_s, :w_s] = src_bgr
        canvas[:h_r, w_s:w_s + w_r] = ref_bgr

        # Step through matches if too many
        step = max(1, len(matches) // max_draw) if len(matches) > max_draw else 1
        sampled = matches[::step]

        for m in sampled:
            sx = int(round(m.get("x", m.get("src_pt", (0, 0))[0])))
            sy = int(round(m.get("y", m.get("src_pt", (0, 0))[1])))
            rx = int(round(m.get("x_ref", m.get("ref_pt", (0, 0))[0]))) + w_s
            ry = int(round(m.get("y_ref", m.get("ref_pt", (0, 0))[1])))

            # Color by confidence: cyan to yellow
            conf = m.get("confidence", 0.8)
            color = (int(255 * (1 - conf)), int(255 * conf), 255)

            cv2.circle(canvas, (sx, sy), 4, color, -1)
            cv2.circle(canvas, (rx, ry), 4, color, -1)
            cv2.line(canvas, (sx, sy), (rx, ry), color, 1, cv2.LINE_AA)

        return canvas

    @staticmethod
    def draw_verification_inliers_outliers(
        img_src: np.ndarray,
        img_ref: np.ndarray,
        all_matches: List[Dict[str, Any]],
        inliers: List[Dict[str, Any]],
        max_draw: int = 150
    ) -> np.ndarray:
        """Visualizes inliers in vibrant green and rejected outliers in red."""
        h_s, w_s = img_src.shape[:2]
        h_r, w_r = img_ref.shape[:2]
        h, w = max(h_s, h_r), w_s + w_r

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        src_bgr = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR) if img_src.ndim == 2 else img_src
        ref_bgr = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR) if img_ref.ndim == 2 else img_ref
        canvas[:h_s, :w_s] = src_bgr
        canvas[:h_r, w_s:w_s + w_r] = ref_bgr

        inlier_set = set()
        for m in inliers:
            sx = round(m.get("x", m.get("src_pt", (0, 0))[0]), 1)
            sy = round(m.get("y", m.get("src_pt", (0, 0))[1]), 1)
            inlier_set.add((sx, sy))

        step = max(1, len(all_matches) // max_draw) if len(all_matches) > max_draw else 1
        for m in all_matches[::step]:
            sx = int(round(m.get("x", m.get("src_pt", (0, 0))[0])))
            sy = int(round(m.get("y", m.get("src_pt", (0, 0))[1])))
            rx = int(round(m.get("x_ref", m.get("ref_pt", (0, 0))[0]))) + w_s
            ry = int(round(m.get("y_ref", m.get("ref_pt", (0, 0))[1])))

            is_inlier = (round(sx, 1), round(sy, 1)) in inlier_set
            color = (0, 230, 70) if is_inlier else (30, 30, 235)  # Green vs Red
            thickness = 2 if is_inlier else 1

            cv2.circle(canvas, (sx, sy), 3, color, -1)
            cv2.circle(canvas, (rx, ry), 3, color, -1)
            cv2.line(canvas, (sx, sy), (rx, ry), color, thickness, cv2.LINE_AA)

        return canvas

    @staticmethod
    def draw_checkerboard_blend(
        img_warped: np.ndarray,
        img_ref: np.ndarray,
        grid_size: int = 64
    ) -> np.ndarray:
        """Generates alternating checkerboard tiles to easily inspect edge alignment."""
        h = min(img_warped.shape[0], img_ref.shape[0])
        w = min(img_warped.shape[1], img_ref.shape[1])

        blend = np.zeros((h, w), dtype=np.uint8)
        w_crop = img_warped[:h, :w]
        r_crop = img_ref[:h, :w]

        for y in range(0, h, grid_size):
            for x in range(0, w, grid_size):
                tile_y = min(h, y + grid_size)
                tile_x = min(w, x + grid_size)

                # Alternate tiles
                if ((y // grid_size) + (x // grid_size)) % 2 == 0:
                    blend[y:tile_y, x:tile_x] = w_crop[y:tile_y, x:tile_x]
                else:
                    blend[y:tile_y, x:tile_x] = r_crop[y:tile_y, x:tile_x]

        return blend

    @staticmethod
    def draw_difference_heatmap(
        img_warped: np.ndarray,
        img_ref: np.ndarray
    ) -> np.ndarray:
        """Visualizes pixel-wise registration residuals as a false-color heatmap."""
        h = min(img_warped.shape[0], img_ref.shape[0])
        w = min(img_warped.shape[1], img_ref.shape[1])

        diff = cv2.absdiff(img_warped[:h, :w], img_ref[:h, :w])
        # Mask out unwarped black borders
        valid_mask = img_warped[:h, :w] > 0
        diff[~valid_mask] = 0

        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_VIRIDIS)
        heatmap[~valid_mask] = [0, 0, 0]
        return heatmap

