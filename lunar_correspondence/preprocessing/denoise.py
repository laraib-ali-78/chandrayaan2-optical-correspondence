"""Edge-preserving denoising for high-resolution lunar imagery.
Adheres to Blueprint Section 6:
'Mild denoise only (edge-preserving) — do not over-smooth,
it erases small craters/boulders that are the main texture cue'
"""

import cv2
import numpy as np


class EdgePreservingDenoise:
    """Edge-preserving filter that maintains lunar micro-craters and rock boulders."""

    def __init__(self, d: int = 5, sigma_color: float = 35.0, sigma_space: float = 35.0):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def filter(self, image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Denoises image while strictly keeping sharp crater boundaries and high gradients intact.
        """
        if method == "bilateral":
            # Bilateral filter preserves sharp edges
            return cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)
        elif method == "median":
            # Removes salt-and-pepper sensor noise
            return cv2.medianBlur(image, 3)
        elif method == "gaussian_mild":
            return cv2.GaussianBlur(image, (3, 3), 0.6)
        else:
            return image

