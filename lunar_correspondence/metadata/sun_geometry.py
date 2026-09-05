"""Solar Illumination Geometry Calculator.
Computes angular illumination differences between acquisition passes.
"""

from typing import Dict, Any, Tuple
import numpy as np


class SunGeometryCalculator:
    """Calculates illumination angle difference between source and reference."""

    @staticmethod
    def calculate_illumination_difference(
        metadata_src: Dict[str, Any],
        metadata_ref: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        """
        Computes azimuth difference, elevation difference, and composite solar angular difference.
        Returns: (composite_illum_diff_deg, delta_azimuth_deg, delta_elevation_deg)
        """
        az_src = metadata_src.get("sun_azimuth_deg", 0.0)
        az_ref = metadata_ref.get("sun_azimuth_deg", 0.0)
        el_src = metadata_src.get("sun_elevation_deg", 45.0)
        el_ref = metadata_ref.get("sun_elevation_deg", 45.0)

        # Angular azimuth difference on circle
        delta_az = abs(az_src - az_ref) % 360.0
        if delta_az > 180.0:
            delta_az = 360.0 - delta_az

        delta_el = abs(el_src - el_ref)

        # 3D unit vectors of sun positions
        # x = cos(el) * cos(az), y = cos(el) * sin(az), z = sin(el)
        v_src = np.array([
            np.cos(np.radians(el_src)) * np.cos(np.radians(az_src)),
            np.cos(np.radians(el_src)) * np.sin(np.radians(az_src)),
            np.sin(np.radians(el_src))
        ])
        v_ref = np.array([
            np.cos(np.radians(el_ref)) * np.cos(np.radians(az_ref)),
            np.cos(np.radians(el_ref)) * np.sin(np.radians(az_ref)),
            np.sin(np.radians(el_ref))
        ])

        dot = np.clip(np.dot(v_src, v_ref), -1.0, 1.0)
        composite_diff = float(np.degrees(np.arccos(dot)))

        return composite_diff, float(delta_az), float(delta_el)

