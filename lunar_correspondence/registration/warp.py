"""Image Warping and GeoTIFF Export.
Implements Stage [14] Image Warping.
"""

from typing import Tuple, Optional
import os
import cv2
import numpy as np


class ImageWarper:
    """Warps source image into reference frame using calculated transformation."""

    @staticmethod
    def warp_source_to_reference(
        img_src: np.ndarray,
        H_src_to_ref: np.ndarray,
        ref_shape: Tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Warps source image so its pixels align with the reference coordinate system.
        """
        h_r, w_r = ref_shape[:2]
        warped = cv2.warpPerspective(
            img_src,
            H_src_to_ref,
            (w_r, h_r),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return warped

    @staticmethod
    def save_registered_image(
        image: np.ndarray,
        output_path: str
    ) -> str:
        """Saves registered raster to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, image)
        return output_path

