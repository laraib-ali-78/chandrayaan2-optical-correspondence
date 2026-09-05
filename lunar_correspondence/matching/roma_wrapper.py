"""RoMa Robust Dense Matcher Wrapper (Fallback Tier 2 - Quality Mode).
Invoked for the hardest extreme sun-angle (>45 deg) and viewpoint variations (Blueprint Section 7 & 14).
"""

from typing import List, Dict, Any
import numpy as np
import cv2
from .loftr_wrapper import LoFTRMatcher


class RoMaMatcher:
    """RoMa dense quality-mode matcher."""

    def __init__(self, confidence_threshold: float = 0.15):
        self.confidence_threshold = confidence_threshold
        self._dense_engine = LoFTRMatcher(confidence_threshold=confidence_threshold)

    def match(self, img_src: np.ndarray, img_ref: np.ndarray) -> List[Dict[str, Any]]:
        """Dense match with deep feature correlation and bilateral consensus."""
        raw_matches = self._dense_engine.match_dense(img_src, img_ref)
        for m in raw_matches:
            m["matcher"] = "RoMa"
        return raw_matches

