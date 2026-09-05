"""Classical SIFT Baseline (Baseline 1).
Implements classical feature detection & description as the benchmark comparison.
Documented to degrade significantly under large sun-angle changes on lunar imagery (MoonMetaSync, arXiv:2410.11118).
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


class SIFTBaseline:
    """Classical SIFT feature detector and descriptor."""

    def __init__(self, nfeatures: int = 2000, contrast_threshold: float = 0.03, edge_threshold: float = 10.0):
        self.nfeatures = nfeatures
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold
        )

    def detect_and_compute(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """Extracts SIFT keypoints and 128-D descriptors."""
        if image.dtype != np.uint8:
            norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = norm.astype(np.uint8)

        kps, desc = self.sift.detectAndCompute(image, mask)
        if desc is None:
            desc = np.empty((0, 128), dtype=np.float32)
        return kps, desc

    @staticmethod
    def match_descriptors(
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = 0.75
    ) -> List[Tuple[int, int, float]]:
        """Lowe's ratio test matching with BFMatcher."""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = matcher.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    # Match score: confidence = 1.0 / (1.0 + m.distance)
                    conf = float(1.0 / (1.0 + max(0.0, m.distance)))
                    good_matches.append((m.queryIdx, m.trainIdx, conf))

        return good_matches

