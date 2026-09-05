"""Photoclinometry / Lunar-Lambert Reflectance (Level 3 Stretch Goal).
Approximates pseudo-albedo using lunar photometric models when DEM is available.
"""

from typing import Optional
import numpy as np


class LunarLambertPhotoclinometry:
    """Photometric pseudo-albedo normalization using Lunar-Lambert law."""

    def __init__(self, L_weight: float = 0.5):
        self.L_weight = L_weight  # Limb-darkening parameter

    def compute_pseudo_albedo(
        self,
        image: np.ndarray,
        incidence_deg: float,
        emission_deg: float = 0.0,
        phase_deg: float = 30.0
    ) -> np.ndarray:
        """
        Applies Lunar-Lambert photometric correction:
        R(i, e, g) = albedo * [2 * L * cos(i) / (cos(i) + cos(e)) + (1 - L) * cos(i)]
        """
        i_rad = np.radians(np.clip(incidence_deg, 0.0, 85.0))
        e_rad = np.radians(np.clip(emission_deg, 0.0, 85.0))

        mu0 = np.cos(i_rad)
        mu = np.cos(e_rad)

        lambert = mu0
        lommel_seeliger = mu0 / (mu0 + mu + 1e-6)

        photometric_factor = 2.0 * self.L_weight * lommel_seeliger + (1.0 - self.L_weight) * lambert
        photometric_factor = max(1e-3, photometric_factor)

        pseudo_albedo = image.astype(np.float32) / photometric_factor
        return np.clip(pseudo_albedo, 0, 255).astype(np.uint8)

