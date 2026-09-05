"""Download and Access Helpers for ISRO PRADAN / ISSDC and NASA LROC.
Implements Section 8 of the Blueprint (manual auth flow, query endpoints, sample cache).
"""

import os
import json
from typing import Dict, Any, List


class LunarDataDownloader:
    """Helper for accessing Chandrayaan-2 and LROC repositories."""

    PRADAN_BASE_URL = "https://pradan.issdc.gov.in/ch2"
    LROC_BASE_URL = "https://wms.lroc.asu.edu/lroc/view_lroc"

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_supported_sensors(self) -> List[Dict[str, Any]]:
        """Returns mission sensor specifications from Blueprint Section 8."""
        return [
            {
                "sensor": "OHRC",
                "portal": "pradan.issdc.gov.in",
                "format": "PDS4 .img + .xml",
                "gsd_m": "0.25 - 0.32",
                "swath": "3 km",
                "notes": "Requires ISSDC/PRADAN account"
            },
            {
                "sensor": "TMC-2",
                "portal": "pradan.issdc.gov.in",
                "format": "PDS4 .img + .xml",
                "gsd_m": "5.0",
                "swath": "20 km",
                "notes": "Includes DEM stereo products"
            },
            {
                "sensor": "IIRS",
                "portal": "pradan.issdc.gov.in",
                "format": "PDS4 .qub",
                "gsd_m": "80.0",
                "swath": "Hyperspectral 256 bands",
                "notes": "Requires PCA panchromatic composite"
            },
            {
                "sensor": "LRO NAC",
                "portal": "lroc.im-ldi.com",
                "format": "PDS3/4 GeoTIFF",
                "gsd_m": "0.5 - 2.0",
                "swath": "5 km",
                "notes": "Publicly accessible reference"
            },
            {
                "sensor": "LRO WAC",
                "portal": "lroc.im-ldi.com",
                "format": "PDS3/4 GeoTIFF",
                "gsd_m": "100.0",
                "swath": "Global mosaic",
                "notes": "Anchor for IIRS multi-hop registration"
            }
        ]

