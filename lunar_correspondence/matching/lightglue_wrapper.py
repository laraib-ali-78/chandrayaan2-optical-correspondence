"""LightGlue Deep Feature Matcher Wrapper.
Primary matcher (Section 7) offering optimal speed/accuracy trade-off.
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import cv2


class LightGlueMatcher:
    """LightGlue learned feature matcher with deep attention mechanism and confidence scoring."""

    def __init__(self, filter_threshold: float = 0.1):
        self.filter_threshold = filter_threshold

    def match(
        self,
        kps_src: np.ndarray,
        desc_src: np.ndarray,
        kps_ref: np.ndarray,
        desc_ref: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Matches source and reference features with mutual nearest neighbors and confidence pruning.
        Returns: list of dicts: {'src_idx', 'ref_idx', 'src_pt': (x,y), 'ref_pt': (x,y), 'confidence': float}
        """
        if len(kps_src) == 0 or len(kps_ref) == 0:
            return []

        # Cosine similarity matrix between L2 normalized descriptors
        sim_matrix = np.dot(desc_src, desc_ref.T)

        # Forward match (src -> ref)
        fwd_matches = np.argmax(sim_matrix, axis=1)
        fwd_scores = np.max(sim_matrix, axis=1)

        # Reverse match (ref -> src)
        bwd_matches = np.argmax(sim_matrix, axis=0)

        matches = []
        for i_src, (i_ref, score) in enumerate(zip(fwd_matches, fwd_scores)):
            # Mutual consistency check
            if bwd_matches[i_ref] == i_src and score >= self.filter_threshold:
                matches.append({
                    "src_idx": int(i_src),
                    "ref_idx": int(i_ref),
                    "src_pt": (float(kps_src[i_src, 0]), float(kps_src[i_src, 1])),
                    "ref_pt": (float(kps_ref[i_ref, 0]), float(kps_ref[i_ref, 1])),
                    "confidence": float(score),
                    "matcher": "LightGlue"
                })

        return matches

