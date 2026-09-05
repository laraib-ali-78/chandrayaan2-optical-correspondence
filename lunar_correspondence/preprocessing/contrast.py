"""Contrast enhancement and radiometric normalization using CLAHE.
Implements Stage [3] Radiometric Preprocessing.
"""

from typing import Tuple
import cv2
import numpy as np


class ContrastNormalizer:
    """CLAHE and dynamic range normalization for lunar imagery."""

    def __init__(self, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Applies CLAHE local contrast normalization."""
        if image.dtype != np.uint8:
            norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image_uint8 = norm.astype(np.uint8)
        else:
            image_uint8 = image

        return self.clahe.apply(image_uint8)

