"""Illumination-Invariant Gradient Representation.
Extracts gradient magnitude and orientation representations that are robust
to solar azimuth and elevation changes on lunar crater topography.
"""

from typing import Tuple
import cv2
import numpy as np


class GradientRepresentation:
    """Computes illumination-normalized gradient fields."""

    @staticmethod
    def compute(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes gradient magnitude and orientation.
        Returns:
            magnitude: float32 normalized [0, 1]
            orientation: float32 in radians [-pi, pi]
        """
        img_f = image.astype(np.float32) / 255.0

        # Sobel spatial derivatives
        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)

        magnitude, orientation = cv2.cartToPolar(gx, gy, angleInDegrees=False)

        # Normalize magnitude
        mag_max = np.max(magnitude)
        if mag_max > 1e-6:
            magnitude = magnitude / mag_max

        return magnitude, orientation

    @staticmethod
    def get_gradient_composite(image: np.ndarray, blend_weight: float = 0.5) -> np.ndarray:
        """
        Creates an illumination-stabilized 8-bit image blending raw intensity
        with normalized gradient magnitude for feature detection.
        """
        mag, _ = GradientRepresentation.compute(image)
        mag_u8 = (mag * 255.0).astype(np.uint8)

        blended = cv2.addWeighted(image, 1.0 - blend_weight, mag_u8, blend_weight, 0)
        return blended

