"""CLI Entry Point & Demonstration Runner for Chandrayaan-2 Pipeline.
Supports running end-to-end registration, 8-stage ablation, test suite, or launching the UI.
"""

import argparse
import sys
import os
import subprocess

from lunar_correspondence.pipeline import LunarRegistrationPipeline
from lunar_correspondence.data_loader.synthetic_generator import LunarSyntheticGenerator
from lunar_correspondence.evaluation.ablation import AblationStudyEngine
from lunar_correspondence.evaluation.report_gen import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Chandrayaan-2 Image Correspondence & Registration (SIH 26166)")
    parser.add_argument("--mode", choices=["demo", "ablation", "test", "ui"], default="demo",
                        help="Execution mode: demo (single pair), ablation (8-stage matrix), test (unit tests), ui (Streamlit app)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "extreme"], default="medium",
                        help="Difficulty bucket for synthetic generation")
    args = parser.parse_args()

    if args.mode == "test":
        print("[TEST] Running Synthetic Regression Test Suite...")
        cmd = [sys.executable, "-m", "unittest", "discover", "-s", "lunar_correspondence/tests/synthetic_regression"]
        res = subprocess.run(cmd)
        sys.exit(res.returncode)

    elif args.mode == "ui":
        print("[UI] Launching Streamlit Interactive UI...")
        cmd = [sys.executable, "-m", "streamlit", "run", "lunar_correspondence/ui/app.py"]
        subprocess.run(cmd)

    elif args.mode == "ablation":
        print(f"[ABLATION] Running 8-Stage Ablation Matrix on '{args.difficulty}' difficulty bucket...")
        generator = LunarSyntheticGenerator(random_seed=42)
        pair = generator.generate_pair(difficulty=args.difficulty)
        ablation_engine = AblationStudyEngine()
        results = ablation_engine.run_full_ablation(pair)
        table = ReportGenerator.generate_ablation_markdown_table(results)
        print("\n" + "=" * 80)
        print("8-STAGE ABLATION RESULTS (Section 20)")
        print("=" * 80)
        print(table)
        print("=" * 80)

    else:  # demo
        print(f"[DEMO] Running End-to-End Registration on '{args.difficulty}' difficulty bucket...")
        generator = LunarSyntheticGenerator(random_seed=42)
        pair = generator.generate_pair(difficulty=args.difficulty)
        pipeline = LunarRegistrationPipeline()

        result = pipeline.process_pair(
            src_raster=pair["source_image"],
            ref_raster=pair["ref_image"],
            meta_src=pair["metadata_src"],
            meta_ref=pair["metadata_ref"],
            H_ground_truth=pair["H_gt"],
            gt_tie_points=pair["gt_tie_points"]
        )

        print("\n" + "=" * 60)
        print("PIPELINE EXPLAINABILITY TRACE:")
        for line in result["pipeline_trace"]:
            print(f"  > {line}")
        print("=" * 60)
        print(f"CONFIDENCE LEVEL: {result['confidence_level']}")
        for r in result['confidence_reasons']:
            print(f"  - {r}")
        print("=" * 60)
        print(f"METRICS SUMMARY:")
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}")
        print("=" * 60)
        print(f"Registered image saved to: {result['registered_image_path']}")


if __name__ == "__main__":
    main()
