"""PDS4 and Planetary Raster Reader for Chandrayaan-2 and LRO.
Handles .xml metadata parsing and .img / raster extraction.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


class PDS4Reader:
    """Reader for Chandrayaan-2 PDS4 XML labels and raster data."""

    def __init__(self):
        pass

    def parse_metadata_xml(self, xml_path: str) -> Dict[str, Any]:
        """Parse PDS4 XML label to extract geometry, sun angles, GSD, and sensor info."""
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"PDS4 XML label not found: {xml_path}")

        metadata = {
            "sensor": "UNKNOWN",
            "acquisition_time": None,
            "sun_azimuth_deg": None,
            "sun_elevation_deg": None,
            "gsd_m_per_px": None,
            "lines": None,
            "samples": None,
            "footprint": [],
            "projection": None,
            "raw_xml_path": xml_path
        }

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Namespaces can vary; search tags ending with specific keywords
            for elem in root.iter():
                tag = elem.tag.split("}")[-1].lower()
                text = (elem.text or "").strip()

                if "instrument_id" in tag or "instrument_name" in tag or "sensor" in tag:
                    if any(s in text.upper() for s in ["OHRC", "TMC", "IIRS", "NAC", "WAC"]):
                        metadata["sensor"] = text.upper()

                elif "start_date_time" in tag or "observation_start_time" in tag:
                    metadata["acquisition_time"] = text

                elif "solar_azimuth" in tag or "sun_azimuth" in tag:
                    try:
                        metadata["sun_azimuth_deg"] = float(text)
                    except ValueError:
                        pass

                elif "solar_elevation" in tag or "sun_elevation" in tag:
                    try:
                        metadata["sun_elevation_deg"] = float(text)
                    except ValueError:
                        pass
                elif "incidence_angle" in tag:
                    # Solar elevation = 90 - incidence angle
                    try:
                        inc = float(text)
                        metadata["sun_elevation_deg"] = 90.0 - inc
                    except ValueError:
                        pass

                elif "pixel_resolution" in tag or "ground_sampling_distance" in tag or "gsd" in tag:
                    try:
                        metadata["gsd_m_per_px"] = float(text)
                    except ValueError:
                        pass

                elif tag == "lines":
                    try:
                        metadata["lines"] = int(text)
                    except ValueError:
                        pass

                elif tag == "samples":
                    try:
                        metadata["samples"] = int(text)
                    except ValueError:
                        pass

                elif "latitude" in tag or "longitude" in tag:
                    try:
                        val = float(text)
                        metadata["footprint"].append(val)
                    except ValueError:
                        pass

        except Exception as e:
            metadata["parse_error"] = str(e)

        # Fallbacks based on sensor defaults if not found in XML
        if metadata["sensor"] != "UNKNOWN" and metadata["gsd_m_per_px"] is None:
            if "OHRC" in metadata["sensor"]:
                metadata["gsd_m_per_px"] = 0.25
            elif "TMC" in metadata["sensor"]:
                metadata["gsd_m_per_px"] = 5.0
            elif "IIRS" in metadata["sensor"]:
                metadata["gsd_m_per_px"] = 80.0
            elif "NAC" in metadata["sensor"]:
                metadata["gsd_m_per_px"] = 0.5
            elif "WAC" in metadata["sensor"]:
                metadata["gsd_m_per_px"] = 100.0

        return metadata

    def load_raster(self, raster_path: str, xml_path: Optional[str] = None) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any], str]:
        """
        Loads raster from image file (.img, .tif, .png, etc.).
        Stage [0] Input Validation:
        Returns: (success: bool, image_array: np.ndarray, metadata: dict, error_reason: str)
        """
        if not os.path.exists(raster_path):
            return False, None, {}, f"File not found: {raster_path}"

        metadata = {}
        if xml_path and os.path.exists(xml_path):
            try:
                metadata = self.parse_metadata_xml(xml_path)
            except Exception as e:
                metadata = {"warning": f"Could not parse XML label: {e}"}
        else:
            # Check if there is an adjacent .xml file
            base_no_ext = os.path.splitext(raster_path)[0]
            candidate_xml = base_no_ext + ".xml"
            if os.path.exists(candidate_xml):
                metadata = self.parse_metadata_xml(candidate_xml)

        # Try reading via OpenCV (supports TIFF, PNG, JPEG, and uncompressed raw formats if header present)
        img = cv2.imread(raster_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            # Try raw binary read if lines & samples are known in metadata
            lines = metadata.get("lines")
            samples = metadata.get("samples")
            if lines and samples:
                try:
                    file_size = os.path.getsize(raster_path)
                    # Test float32, uint16, uint8
                    dtype = np.uint8
                    if file_size >= lines * samples * 2:
                        dtype = np.uint16
                    elif file_size >= lines * samples * 4:
                        dtype = np.float32
                    
                    with open(raster_path, "rb") as f:
                        f.seek(file_size - lines * samples * np.dtype(dtype).itemsize)
                        raw_data = np.fromfile(f, dtype=dtype)
                        img = raw_data.reshape((lines, samples))
                except Exception as e:
                    return False, None, metadata, f"Corrupt or unsupported raw raster: {e}"
            else:
                return False, None, metadata, f"Unreadable raster format and missing XML dimensions: {raster_path}"

        # Normalize to 2D uint8/float32 grayscale
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Ensure normalized range
        if img.dtype != np.uint8:
            img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img_norm.astype(np.uint8)

        metadata["shape"] = img.shape
        if metadata.get("gsd_m_per_px") is None:
            # Default fallback GSD
            metadata["gsd_m_per_px"] = 1.0

        return True, img, metadata, "OK"

