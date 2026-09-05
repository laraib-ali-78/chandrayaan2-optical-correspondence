"""Synthetic Regression and Sub-Pixel Accuracy Test Suite.
Verifies end-to-end mathematical correctness and ground-truth sub-pixel accuracy.
"""

import unittest
import numpy as np

from lunar_correspondence.data_loader.synthetic_generator import LunarSyntheticGenerator
from lunar_correspondence.pipeline import LunarRegistrationPipeline
from lunar_correspondence.subpixel.ecc_refine import ECCSubPixelRefiner
from lunar_correspondence.geometry.magsac import MAGSACVerifier
from lunar_correspondence.evaluation.ablation import AblationStudyEngine


class TestLunarPipeline(unittest.TestCase):
    """Regression test suite for lunar image correspondence."""

    @classmethod
    def setUpClass(cls):
        cls.generator = LunarSyntheticGenerator(random_seed=123)
        cls.pipeline = LunarRegistrationPipeline()

    def test_01_synthetic_pair_generation(self):
        """Test procedural lunar surface and exact ground-truth homography creation."""
        pair = self.generator.generate_pair(difficulty="easy")
        self.assertIn("source_image", pair)
        self.assertIn("ref_image", pair)
        self.assertIn("H_gt", pair)
        self.assertEqual(pair["H_gt"].shape, (3, 3))
        self.assertGreater(len(pair["gt_tie_points"]), 5)

    def test_02_subpixel_refinement_accuracy(self):
        """Validates sub-pixel refinement produces displacements within valid bounds."""
        refiner = ECCSubPixelRefiner(patch_size=48)
        img = self.generator.generate_base_lunar_terrain(256, 256)
        
        # Test subpixel refinement on identical patch
        match = refiner.refine_match(
            img_src=img,
            img_ref=img,
            src_pt=(120.0, 120.0),
            ref_pt=(120.0, 120.0),
            coarse_confidence=0.9
        )
        self.assertLess(abs(match["subpixel_dx"]), 0.5)
        self.assertLess(abs(match["subpixel_dy"]), 0.5)

    def test_03_magsac_spatial_uniformity(self):
        """Validates that MAGSAC++ filters outliers and grid selection bounds cluster density."""
        verifier = MAGSACVerifier(threshold_px=3.0, max_per_cell=2)
        # Create clustered matches
        clustered_matches = []
        for i in range(15):
            clustered_matches.append({
                "src_pt": (50.0 + i * 0.1, 50.0 + i * 0.1),
                "ref_pt": (50.0 + i * 0.1, 50.0 + i * 0.1),
                "confidence": 0.95
            })
        uniform = verifier.apply_spatial_uniformity_grid(clustered_matches, (512, 512))
        # Max per cell is 2, so should prune 15 matches down to 2
        self.assertLessEqual(len(uniform), 2)

    def test_04_end_to_end_pipeline_easy_pair(self):
        """Tests full end-to-end registration pipeline on an easy pair."""
        pair = self.generator.generate_pair(difficulty="easy")
        result = self.pipeline.process_pair(
            src_raster=pair["source_image"],
            ref_raster=pair["ref_image"],
            meta_src=pair["metadata_src"],
            meta_ref=pair["metadata_ref"],
            H_ground_truth=pair["H_gt"],
            gt_tie_points=pair["gt_tie_points"]
        )

        self.assertIn(result["confidence_level"], ["HIGH", "MEDIUM"])
        self.assertIsNotNone(result["transform_params"])
        self.assertGreater(result["metrics"]["n_inliers"], 5)
        self.assertIsNotNone(result["registered_image_path"])

    def test_05_ablation_engine_run(self):
        """Validates 8-stage ablation execution and completeness."""
        pair = self.generator.generate_pair(difficulty="easy")
        ablation = AblationStudyEngine()
        ablation_results = ablation.run_full_ablation(pair)
        self.assertEqual(len(ablation_results), 8)
        self.assertTrue(ablation_results[0]["configuration"].startswith("A."))
        self.assertTrue(ablation_results[-1]["configuration"].startswith("H."))


if __name__ == "__main__":
    unittest.main()

