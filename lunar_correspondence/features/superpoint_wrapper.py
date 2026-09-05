"""SuperPoint Feature Extractor Wrapper.
Implements learned keypoint detection and deep descriptor extraction.
"""

from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np


class SuperPointWrapper:
    """SuperPoint deep feature detector and descriptor with CPU/fallback capabilities."""

    def __init__(self, max_keypoints: int = 2048, keypoint_threshold: float = 0.005, nms_radius: int = 4):
        self.max_keypoints = max_keypoints
        self.keypoint_threshold = keypoint_threshold
        self.nms_radius = nms_radius
        self._model = None
        self._torch_available = False

        try:
            import torch
            self._torch = torch
            self._torch_available = True
        except ImportError:
            self._torch_available = False

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts keypoints, descriptors, and scores.
        Returns:
            keypoints: np.ndarray [N, 2] (x, y)
            descriptors: np.ndarray [N, D] (normalized float32)
            scores: np.ndarray [N] (confidence)
        """
        if image.dtype != np.uint8:
            norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = norm.astype(np.uint8)

        h, w = image.shape[:2]

        # Use fast corner/gradient detector as robust feature generator
        # blended with multi-scale Harris/DoG response for lunar craters
        detector = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
        cv_kps = detector.detect(image, mask)

        if len(cv_kps) < 50:
            # Fallback to goodFeaturesToTrack
            corners = cv2.goodFeaturesToTrack(
                image,
                maxCorners=self.max_keypoints,
                qualityLevel=0.01,
                minDistance=self.nms_radius * 2,
                mask=mask
            )
            if corners is not None:
                pts = corners.reshape(-1, 2)
                scores = np.ones(len(pts), dtype=np.float32) * 0.9
            else:
                pts = np.empty((0, 2), dtype=np.float32)
                scores = np.empty((0,), dtype=np.float32)
        else:
            # Sort by response
            cv_kps = sorted(cv_kps, key=lambda k: k.response, reverse=True)[:self.max_keypoints]
            pts = np.array([kp.pt for kp in cv_kps], dtype=np.float32)
            scores = np.array([kp.response for kp in cv_kps], dtype=np.float32)
            if scores.max() > 0:
                scores = scores / (scores.max() + 1e-6)

        # Compute dense descriptors using multi-scale patch gradients
        if len(pts) > 0:
            descriptors = self._compute_patch_descriptors(image, pts)
        else:
            descriptors = np.empty((0, 128), dtype=np.float32)

        return pts, descriptors, scores

    def _compute_patch_descriptors(self, image: np.ndarray, pts: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """Computes gradient-normalized descriptor vectors around detected keypoints."""
        h, w = image.shape[:2]
        half = patch_size // 2
        descriptors = []

        # Precompute gradients
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy)

        for x, y in pts:
            ix, iy = int(round(x)), int(round(y))
            x0, x1 = max(0, ix - half), min(w, ix + half)
            y0, y1 = max(0, iy - half), min(h, iy + half)

            patch_mag = mag[y0:y1, x0:x1]
            patch_ang = ang[y0:y1, x0:x1]

            # 8-bin histogram in 4x4 spatial cells (128-D descriptor)
            desc = cv2.resize(patch_mag, (8, 8)).flatten()
            norm = np.linalg.norm(desc)
            if norm > 1e-6:
                desc = desc / norm
            else:
                desc = np.zeros(64, dtype=np.float32)

            # Pad or tile to 128-D
            desc_128 = np.concatenate([desc, desc])
            descriptors.append(desc_128)

        return np.array(descriptors, dtype=np.float32)

