"""Synthetic Lunar Image & Ground-Truth Pair Generator.
Implements Section 10 & 11 of the Blueprint:
1. Procedural generation of realistic lunar crater surfaces (micro-craters, boulders, crater rims).
2. Known geometric transformations (Affine, Similarity, Homography with exact inverse).
3. Photometric perturbations (sun-angle illumination gradient, directional shadows, sensor noise).
4. Difficulty taxonomy generation: Easy, Medium, Hard, Extreme.
5. Exact ground-truth tie-point pairs for sub-pixel accuracy verification.
"""

import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import cv2


class LunarSyntheticGenerator:
    """Generates synthetic lunar terrain and ground-truth pairs with mathematically exact transforms."""

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def generate_base_lunar_terrain(self, width: int = 1024, height: int = 1024, num_craters: int = 80) -> np.ndarray:
        """Procedurally synthesizes realistic lunar crater terrain with fractal noise and impact craters."""
        # 1. Base regolith albedo texture
        base = self.rng.normal(120, 15, (height, width)).astype(np.float32)
        base = cv2.GaussianBlur(base, (7, 7), 2.0)

        # 2. Multi-scale craters
        for _ in range(num_craters):
            cx = self.rng.randint(20, width - 20)
            cy = self.rng.randint(20, height - 20)
            radius = self.rng.randint(8, 90)
            depth = self.rng.uniform(25, 75)

            y, x = np.ogrid[:height, :width]
            dist_sq = (x - cx) ** 2 + (y - cy) ** 2
            dist = np.sqrt(dist_sq)

            mask = dist <= radius * 1.4
            # Crater bowl
            bowl = np.maximum(0.0, 1.0 - (dist / radius) ** 2) * depth
            # Crater rim
            rim = np.exp(-((dist - radius) ** 2) / (2 * (radius * 0.2) ** 2)) * (depth * 0.45)

            base[mask] = np.clip(base[mask] - bowl[mask] + rim[mask], 10, 245)

        # 3. Small boulders and rocks
        num_boulders = self.rng.randint(150, 300)
        for _ in range(num_boulders):
            bx = self.rng.randint(5, width - 5)
            by = self.rng.randint(5, height - 5)
            br = self.rng.randint(2, 5)
            cv2.circle(base, (bx, by), br, float(self.rng.randint(180, 255)), -1)

        terrain = np.clip(base, 0, 255).astype(np.uint8)
        return terrain

    def apply_illumination_and_shadows(
        self,
        image: np.ndarray,
        sun_azimuth_deg: float,
        sun_elevation_deg: float,
        shadow_intensity: float = 0.6
    ) -> np.ndarray:
        """
        Simulates directional solar illumination and shadows based on lunar photometry.
        Sun elevation determines shadow length; azimuth determines shadow angle.
        """
        h, w = image.shape
        # Solar vector
        az_rad = np.radians(sun_azimuth_deg)
        el_rad = np.radians(max(sun_elevation_deg, 3.0))

        sun_dx = np.cos(az_rad)
        sun_dy = np.sin(az_rad)

        # Compute terrain gradient (surface normals)
        sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

        # Cosine of incidence angle
        shading = (sobel_x * sun_dx + sobel_y * sun_dy) * (1.0 / np.tan(el_rad)) * 0.15

        # Illumination gradient across the frame
        xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        global_gradient = (xx * sun_dx + yy * sun_dy) * 20.0

        perturbed = image.astype(np.float32) + shading + global_gradient

        # Add sharp shadow casting for low sun elevation
        if sun_elevation_deg < 25.0:
            shadow_mask = shading < -15.0
            perturbed[shadow_mask] *= (1.0 - shadow_intensity)

        return np.clip(perturbed, 0, 255).astype(np.uint8)

    def generate_pair(
        self,
        base_image: Optional[np.ndarray] = None,
        difficulty: str = "easy",
        source_sensor: str = "OHRC",
        ref_sensor: str = "LRO_NAC",
        image_size: Tuple[int, int] = (640, 640)
    ) -> Dict[str, Any]:
        """
        Generates a ground-truth image pair with known transformation matrix and difficulty parameters.
        Returns:
            {
                'source_image': np.ndarray,
                'ref_image': np.ndarray,
                'H_gt': np.ndarray (3x3 ground truth homography mapping source -> ref),
                'H_inv': np.ndarray (3x3 inverse mapping ref -> source),
                'gt_tie_points': list of ((x_src, y_src), (x_ref, y_ref)),
                'metadata_src': dict,
                'metadata_ref': dict,
                'difficulty': str
            }
        """
        w, h = image_size
        if base_image is None:
            canvas_w, canvas_h = int(w * 1.8), int(h * 1.8)
            base_canvas = self.generate_base_lunar_terrain(canvas_w, canvas_h)
        else:
            base_canvas = base_image

        # Difficulty taxonomy parameters (Blueprint Section 11)
        if difficulty == "easy":
            sun_diff = self.rng.uniform(2.0, 12.0)
            angle_deg = self.rng.uniform(-8.0, 8.0)
            scale = self.rng.uniform(0.95, 1.05)
            noise_sigma = 2.0
            src_sun_el = 45.0
            ref_sun_el = 45.0 + self.rng.uniform(-3, 3)
            src_sun_az = 60.0
            ref_sun_az = src_sun_az + sun_diff
            perspective_strength = 0.0001
        elif difficulty == "medium":
            sun_diff = self.rng.uniform(18.0, 38.0)
            angle_deg = self.rng.uniform(-25.0, 25.0)
            scale = self.rng.uniform(0.85, 1.20)
            noise_sigma = 5.0
            src_sun_el = 35.0
            ref_sun_el = 25.0
            src_sun_az = 40.0
            ref_sun_az = src_sun_az + sun_diff
            perspective_strength = 0.0004
        elif difficulty == "hard":
            sun_diff = self.rng.uniform(48.0, 75.0)
            angle_deg = self.rng.uniform(-45.0, 45.0)
            scale = self.rng.uniform(0.70, 1.40)
            noise_sigma = 8.0
            src_sun_el = 15.0
            ref_sun_el = 50.0
            src_sun_az = 30.0
            ref_sun_az = src_sun_az + sun_diff
            perspective_strength = 0.0008
        else:  # extreme
            sun_diff = self.rng.uniform(80.0, 120.0)
            angle_deg = self.rng.uniform(-70.0, 70.0)
            scale = self.rng.uniform(0.5, 1.8)
            noise_sigma = 12.0
            src_sun_el = 4.5  # near terminator
            ref_sun_el = 60.0
            src_sun_az = 15.0
            ref_sun_az = src_sun_az + sun_diff
            perspective_strength = 0.0012

        # Crop reference image from canvas
        cx, cy = base_canvas.shape[1] // 2, base_canvas.shape[0] // 2
        ref_x0 = max(0, cx - w // 2)
        ref_y0 = max(0, cy - h // 2)
        ref_crop = base_canvas[ref_y0:ref_y0 + h, ref_x0:ref_x0 + w].copy()

        # Build ground-truth Homography: Center -> Rotate/Scale/Perspective -> Translation
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        tx = self.rng.uniform(-25, 25)
        ty = self.rng.uniform(-25, 25)

        # 3x3 Projective Homography
        H_center = np.array([
            [1.0, 0.0, -w / 2.0],
            [0.0, 1.0, -h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        H_transf = np.array([
            [scale * cos_a, -scale * sin_a, tx],
            [scale * sin_a, scale * cos_a, ty],
            [perspective_strength, -perspective_strength * 0.5, 1.0]
        ], dtype=np.float64)

        H_uncenter = np.array([
            [1.0, 0.0, w / 2.0],
            [0.0, 1.0, h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # H maps Reference coords to Source coords: p_src = H * p_ref
        H_ref_to_src = H_uncenter @ H_transf @ H_center
        # H_src_to_ref is the registration target: p_ref = H_gt * p_src
        H_gt = np.linalg.inv(H_ref_to_src)

        # Warp reference patch to create source patch
        source_crop = cv2.warpPerspective(ref_crop, H_ref_to_src, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Apply realistic illumination & sensor noise
        source_perturbed = self.apply_illumination_and_shadows(source_crop, src_sun_az, src_sun_el)
        ref_perturbed = self.apply_illumination_and_shadows(ref_crop, ref_sun_az, ref_sun_el)

        if noise_sigma > 0:
            noise_src = self.rng.normal(0, noise_sigma, source_perturbed.shape)
            noise_ref = self.rng.normal(0, noise_sigma, ref_perturbed.shape)
            source_perturbed = np.clip(source_perturbed + noise_src, 0, 255).astype(np.uint8)
            ref_perturbed = np.clip(ref_perturbed + noise_ref, 0, 255).astype(np.uint8)

        # Generate exact ground-truth tie points
        gt_tie_points = []
        grid_coords = np.linspace(80, min(w, h) - 80, 8)
        for gy in grid_coords:
            for gx in grid_coords:
                pt_src_homo = np.array([gx, gy, 1.0], dtype=np.float64)
                pt_ref_homo = H_gt @ pt_src_homo
                pt_ref_x = pt_ref_homo[0] / pt_ref_homo[2]
                pt_ref_y = pt_ref_homo[1] / pt_ref_homo[2]
                if 10 <= pt_ref_x < w - 10 and 10 <= pt_ref_y < h - 10:
                    gt_tie_points.append(((float(gx), float(gy)), (float(pt_ref_x), float(pt_ref_y))))

        metadata_src = {
            "sensor": source_sensor,
            "gsd_m_per_px": 0.25 if "OHRC" in source_sensor else 5.0,
            "sun_azimuth_deg": float(src_sun_az),
            "sun_elevation_deg": float(src_sun_el),
            "acquisition_time": "2024-02-14T10:30:00Z",
            "lines": h,
            "samples": w
        }

        metadata_ref = {
            "sensor": ref_sensor,
            "gsd_m_per_px": 0.5 if "NAC" in ref_sensor else 100.0,
            "sun_azimuth_deg": float(ref_sun_az),
            "sun_elevation_deg": float(ref_sun_el),
            "acquisition_time": "2024-03-21T14:15:00Z",
            "lines": h,
            "samples": w
        }

        return {
            "source_image": source_perturbed,
            "ref_image": ref_perturbed,
            "H_gt": H_gt,
            "H_inv": H_ref_to_src,
            "gt_tie_points": gt_tie_points,
            "metadata_src": metadata_src,
            "metadata_ref": metadata_ref,
            "difficulty": difficulty
        }

