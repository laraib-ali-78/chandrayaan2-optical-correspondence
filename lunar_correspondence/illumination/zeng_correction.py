"""Zeng-Style Illumination-Orientation Correction.
Normalizes local descriptor orientations relative to the illumination azimuth,
satisfying Section 12 (Level 2) and Section 20 (Ablation B).
"""

from typing import List, Tuple
import cv2
import numpy as np


class ZengIlluminationCorrection:
    """Corrects keypoint descriptor orientations based on dominant solar azimuth."""

    def __init__(self, sun_azimuth_deg: float = 0.0):
        self.sun_azimuth_deg = sun_azimuth_deg

    def correct_keypoints_orientation(
        self,
        keypoints: List[cv2.KeyPoint],
        reference_azimuth_deg: float = 0.0
    ) -> List[cv2.KeyPoint]:
        """
        Rotates keypoint canonical orientations by the delta between acquisition solar azimuths.
        Delta = (sun_azimuth - reference_azimuth)
        """
        delta_az = (self.sun_azimuth_deg - reference_azimuth_deg) % 360.0

        corrected_kps = []
        for kp in keypoints:
            new_angle = (kp.angle - delta_az) % 360.0
            corrected_kp = cv2.KeyPoint(
                x=kp.pt[0],
                y=kp.pt[1],
                size=kp.size,
                angle=new_angle,
                response=kp.response,
                octave=kp.octave,
                class_id=kp.class_id
            )
            corrected_kps.append(corrected_kp)

        return corrected_kps

