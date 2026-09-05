"""Radiometric calibration for lunar optical sensors.
Converts raw Digital Numbers (DN) to radiance/reflectance (I/F).
"""

from typing import Optional, Dict, Any
import numpy as np


class RadiometricCalibrator:
    """Applies sensor calibration coefficients if available in metadata."""

    def calibrate(self, raw_img: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """
        Converts DN to calibrated reflectance (I/F) or normalized radiance.
        Formula: Radiance = (DN - Offset) * Gain
        """
        gain = metadata.get("gain", 1.0)
        offset = metadata.get("offset", 0.0)

        # If already calibrated or default
        if gain == 1.0 and offset == 0.0:
            return raw_img

        calibrated = (raw_img.astype(np.float32) - offset) * gain
        calibrated = np.clip(calibrated, 0.0, None)
        return calibrated

