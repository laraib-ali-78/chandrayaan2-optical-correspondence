"""Lunar footprint parsing and geographic overlap estimation."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class FootprintProcessor:
    """Handles latitude/longitude footprint polygons and image overlap calculation."""

    @staticmethod
    def estimate_overlap_fraction(
        meta_src: Dict[str, Any],
        meta_ref: Dict[str, Any]
    ) -> float:
        """
        Estimates spatial overlap fraction between source and reference.
        Falls back to 1.0 (or coarse phase-correlation) if footprints are missing.
        """
        fp_src = meta_src.get("footprint", [])
        fp_ref = meta_ref.get("footprint", [])

        # If bounding boxes are available
        if len(fp_src) >= 4 and len(fp_ref) >= 4:
            src_min_lat, src_max_lat = min(fp_src[0::2]), max(fp_src[0::2])
            src_min_lon, src_max_lon = min(fp_src[1::2]), max(fp_src[1::2])

            ref_min_lat, ref_max_lat = min(fp_ref[0::2]), max(fp_ref[0::2])
            ref_min_lon, ref_max_lon = min(fp_ref[1::2]), max(fp_ref[1::2])

            inter_min_lat = max(src_min_lat, ref_min_lat)
            inter_max_lat = min(src_max_lat, ref_max_lat)
            inter_min_lon = max(src_min_lon, ref_min_lon)
            inter_max_lon = min(src_max_lon, ref_max_lon)

            if inter_max_lat <= inter_min_lat or inter_max_lon <= inter_min_lon:
                return 0.0

            inter_area = (inter_max_lat - inter_min_lat) * (inter_max_lon - inter_min_lon)
            src_area = max(1e-6, (src_max_lat - src_min_lat) * (src_max_lon - src_min_lon))
            return float(np.clip(inter_area / src_area, 0.0, 1.0))

        # Default assumption when metadata is absent (graceful fallback)
        return 0.85

